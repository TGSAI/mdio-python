"""Conversion from to MDIO various other formats."""


from __future__ import annotations

from os import path
from tempfile import TemporaryDirectory

import dask.array as da
import numpy as np
from tqdm.auto import tqdm

from mdio import MDIOReader
from mdio.segy.byte_utils import ByteOrder
from mdio.segy.byte_utils import Dtype
from mdio.segy.creation import concat_files
from mdio.segy.creation import mdio_spec_to_segy
from mdio.segy.creation import prepare_headers
from mdio.segy.creation import prepare_samples
from mdio.segy.creation import write_to_segy_stack


try:
    import distributed
except ImportError:
    distributed = None


def mdio_to_segy(  # noqa: C901
    mdio_path_or_buffer: str,
    output_segy_path: str,
    endian: str = "big",
    access_pattern: str = "012",
    out_sample_format: str = "ibm32",
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
        out_sample_format: Output sample format.
            Currently support: {'ibm32', 'float32'}. Default is 'ibm32'.
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
        ...     out_sample_format="float32",
        ... )

    """
    backend = "dask"

    mdio = MDIOReader(
        mdio_path_or_buffer=mdio_path_or_buffer,
        access_pattern=access_pattern,
        storage_options=storage_options,
    )

    ndim = mdio.n_dim

    # We flatten the z-axis (time or depth); so ieee2ibm, and byte-swaps etc
    # can run on big chunks of data.
    auto_chunk = (None,) * (ndim - 1) + ("100M",)
    new_chunks = new_chunks if new_chunks is not None else auto_chunk

    creation_args = [
        mdio_path_or_buffer,
        output_segy_path,
        endian,
        access_pattern,
        out_sample_format,
        storage_options,
        new_chunks,
        selection_mask,
        backend,
    ]

    if client is not None:
        if distributed is not None:
            # This is in case we work with big data
            feature = client.submit(mdio_spec_to_segy, *creation_args)
            mdio, sample_format = feature.result()
        else:
            msg = "Distributed client was provided, but `distributed` is not installed"
            raise ImportError(msg)
    else:
        mdio, sample_format = mdio_spec_to_segy(*creation_args)

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

    # Parse output type and byte order
    out_dtype = Dtype[out_sample_format.upper()]
    out_byteorder = ByteOrder[endian.upper()]

    samples_proc = samples.map_blocks(
        prepare_samples,
        out_dtype=out_dtype,
        out_byteorder=out_byteorder,
    )
    headers_proc = headers.map_blocks(
        prepare_headers,
        out_byteorder=out_byteorder,
    )

    trace_dtype = {
        "names": ("header", "pad", "trace"),
        "formats": [
            headers_proc.dtype,
            np.dtype("int64"),
            samples_proc.shape[-1] * samples_proc.dtype,
        ],
    }

    trace_dtype = np.dtype(trace_dtype)

    # tmp file root
    out_dir = path.dirname(output_segy_path)
    tmp_dir = TemporaryDirectory(dir=out_dir)

    lazy_traces = da.map_blocks(
        write_to_segy_stack,
        samples_proc,
        headers_proc[..., None],
        live_mask[..., None],
        file_root=tmp_dir.name,
        trace_dtype=trace_dtype,
        drop_axis=-1,
    )

    tqdm_kw = dict(
        desc="Writing Blocks",
        total=lazy_traces.blocks.shape[0],
        unit="block",
        dynamic_ncols=True,
    )
    with tmp_dir:
        for segy_block in tqdm(lazy_traces.blocks, **tqdm_kw):
            partial_files = segy_block.compute()

            concat_file_paths = [output_segy_path]

            partial_list = partial_files.ravel().tolist()
            partial_list = [path.join(tmp_dir.name, file) for file in partial_list]
            partial_list.sort()

            concat_file_paths += partial_list

            if client is not None:
                _ = client.submit(concat_files, concat_file_paths).result()
            else:
                concat_files(concat_file_paths)
