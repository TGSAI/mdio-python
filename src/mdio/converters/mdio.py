"""Conversion from to MDIO various other formats."""

from __future__ import annotations

import os
from os import path
from tempfile import TemporaryDirectory

import numpy as np
from psutil import cpu_count
from tqdm.dask import TqdmCallback

from mdio import MDIOReader
from mdio.segy.blocked_io import to_segy
from mdio.segy.creation import concat_files
from mdio.segy.creation import mdio_spec_to_segy
from mdio.segy.utilities import segy_export_rechunker


try:
    import distributed
except ImportError:
    distributed = None


default_cpus = cpu_count(logical=True)
NUM_CPUS = int(os.getenv("MDIO__EXPORT__CPU_COUNT", default_cpus))


def mdio_to_segy(  # noqa: C901
    mdio_path_or_buffer: str,
    output_segy_path: str,
    endian: str = "big",
    access_pattern: str = "012",
    storage_options: dict = None,
    new_chunks: tuple[int, ...] = None,
    selection_mask: np.ndarray = None,
    client: distributed.Client = None,
) -> None:
    """Convert MDIO file to SEG-Y format.

    MDIO allows exporting multidimensional seismic data back to the flattened
    seismic format SEG-Y, to be used in data transmission.

    The input headers are preserved as is, and will be transferred to the
    output file.

    The user has control over the endianness, and the floating point data
    type. However, by default we export as Big-Endian IBM float, per the
    SEG-Y format's default.

    The input MDIO can be local or cloud based. However, the output SEG-Y
    will be generated locally.

    A `selection_mask` can be provided (in the shape of the spatial grid)
    to export a subset of the seismic data.

    Args:
        mdio_path_or_buffer: Input path where the MDIO is located
        output_segy_path: Path to the output SEG-Y file
        endian: Endianness of the input SEG-Y. Rev.2 allows little
            endian. Default is 'big'.
        access_pattern: This specificies the chunk access pattern. Underlying
            zarr.Array must exist. Examples: '012', '01'
        storage_options: Storage options for the cloud storage backend.
            Default: None (will assume anonymous access)
        new_chunks: Set manual chunksize. For development purposes only.
        selection_mask: Array that lists the subset of traces
        client: Dask client. If `None` we will use local threaded scheduler.
            If `auto` is used we will create multiple processes (with
            8 threads each).

    Raises:
        ImportError: if distributed package isn't installed but requested.
        ValueError: if cut mask is empty, i.e. no traces will be written.

    Examples:
        To export an existing local MDIO file to SEG-Y we use the code
        snippet below. This will export the full MDIO (without padding) to
        SEG-Y format using IBM floats and big-endian byte order.

        >>> from mdio import mdio_to_segy
        >>>
        >>>
        >>> mdio_to_segy(
        ...     mdio_path_or_buffer="prefix2/file.mdio",
        ...     output_segy_path="prefix/file.segy",
        ... )

        If we want to export this as an IEEE big-endian, using a selection
        mask, we would run:

        >>> mdio_to_segy(
        ...     mdio_path_or_buffer="prefix2/file.mdio",
        ...     output_segy_path="prefix/file.segy",
        ...     selection_mask=boolean_mask,
        ... )

    """
    backend = "dask"

    mdio = MDIOReader(
        mdio_path_or_buffer=mdio_path_or_buffer,
        access_pattern=access_pattern,
        storage_options=storage_options,
    )

    if new_chunks is None:
        new_chunks = segy_export_rechunker(mdio.chunks, mdio.shape, mdio._traces.dtype)

    creation_args = [
        mdio_path_or_buffer,
        output_segy_path,
        access_pattern,
        endian,
        storage_options,
        new_chunks,
        backend,
    ]

    if client is not None:
        if distributed is not None:
            # This is in case we work with big data
            feature = client.submit(mdio_spec_to_segy, *creation_args)
            mdio, segy_factory = feature.result()
        else:
            msg = "Distributed client was provided, but `distributed` is not installed"
            raise ImportError(msg)
    else:
        mdio, segy_factory = mdio_spec_to_segy(*creation_args)

    live_mask = mdio.live_mask.compute()

    if selection_mask is not None:
        live_mask = live_mask & selection_mask

    # This handles the case if we are skipping a whole block.
    if live_mask.sum() == 0:
        raise ValueError("No traces will be written out. Live mask is empty.")

    # Find rough dim limits, so we don't unnecessarily hit disk / cloud store.
    # Typically, gets triggered when there is a selection mask
    dim_slices = ()
    live_nonzeros = live_mask.nonzero()
    for dim_nonzeros in live_nonzeros:
        start = np.min(dim_nonzeros)
        stop = np.max(dim_nonzeros) + 1
        dim_slices += (slice(start, stop),)

    # Lazily pull the data with limits now, and limit mask so its the same shape.
    live_mask, headers, samples = mdio[dim_slices]
    live_mask = live_mask.rechunk(headers.chunksize)

    if selection_mask is not None:
        selection_mask = selection_mask[dim_slices]
        live_mask = live_mask & selection_mask

    # tmp file root
    out_dir = path.dirname(output_segy_path)
    tmp_dir = TemporaryDirectory(dir=out_dir)

    with tmp_dir:
        with TqdmCallback(desc="Unwrapping MDIO Blocks"):
            flat_files = to_segy(
                samples=samples,
                headers=headers,
                live_mask=live_mask,
                segy_factory=segy_factory,
                file_root=tmp_dir.name,
                axis=tuple(range(1, samples.ndim)),
            )

            if client is not None:
                flat_files = flat_files.compute()
            else:
                flat_files = flat_files.compute(num_workers=NUM_CPUS)

        # If whole blocks are missing, remove them from the list.
        missing_mask = flat_files == "missing"
        flat_files = flat_files[~missing_mask]

        final_concat = [output_segy_path] + flat_files.tolist()

        if client is not None:
            _ = client.submit(concat_files, final_concat).result()
        else:
            concat_files(final_concat, progress=True)
