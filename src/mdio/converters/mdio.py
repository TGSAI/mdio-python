"""Conversion from to MDIO various other formats."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import dask.array as da
import numpy as np
import xarray as xr
from psutil import cpu_count
from tqdm.dask import TqdmCallback

from mdio import MDIOReader
from mdio.core.storage_location import StorageLocation
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


def mdio_to_segy(  # noqa: PLR0912, PLR0913
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

    We export N-D seismic data to the flattened SEG-Y format used in data transmission.

    The input headers are preserved as is, and will be transferred to the output file.

    Input MDIO can be local or cloud based. However, the output SEG-Y will be generated locally.

    A `selection_mask` can be provided (same shape as spatial grid) to export a subset.

    Args:
        mdio_path_or_buffer: Input path where the MDIO is located.
        output_segy_path: Path to the output SEG-Y file.
        endian: Endianness of the input SEG-Y. Rev.2 allows little endian. Default is 'big'.
        access_pattern: This specificies the chunk access pattern. Underlying zarr.Array must
            exist. Examples: '012', '01'
        storage_options: Storage options for the cloud storage backend. Default: None (anonymous)
        new_chunks: Set manual chunksize. For development purposes only.
        selection_mask: Array that lists the subset of traces
        client: Dask client. If `None` we will use local threaded scheduler. If `auto` is used we
            will create multiple processes (with 8 threads each).

    Raises:
        ImportError: if distributed package isn't installed but requested.
        ValueError: if cut mask is empty, i.e. no traces will be written.

    Examples:
        To export an existing local MDIO file to SEG-Y we use the code snippet below. This will
        export the full MDIO (without padding) to SEG-Y format using IBM floats and big-endian
        byte order.

        >>> from mdio import mdio_to_segy
        >>>
        >>>
        >>> mdio_to_segy(
        ...     mdio_path_or_buffer="prefix2/file.mdio",
        ...     output_segy_path="prefix/file.segy",
        ... )

        If we want to export this as an IEEE big-endian, using a selection mask, we would run:

        >>> mdio_to_segy(
        ...     mdio_path_or_buffer="prefix2/file.mdio",
        ...     output_segy_path="prefix/file.segy",
        ...     selection_mask=boolean_mask,
        ... )

    """
    backend = "dask"

    output_segy_path = Path(output_segy_path)

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
        msg = "No traces will be written out. Live mask is empty."
        raise ValueError(msg)

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
    live_mask = live_mask.rechunk(headers.chunks)

    if selection_mask is not None:
        selection_mask = selection_mask[dim_slices]
        live_mask = live_mask & selection_mask

    # tmp file root
    out_dir = output_segy_path.parent
    tmp_dir = TemporaryDirectory(dir=out_dir)

    with tmp_dir:
        with TqdmCallback(desc="Unwrapping MDIO Blocks"):
            block_records = to_segy(
                samples=samples,
                headers=headers,
                live_mask=live_mask,
                segy_factory=segy_factory,
                file_root=tmp_dir.name,
            )

            if client is not None:
                block_records = block_records.compute()
            else:
                block_records = block_records.compute(num_workers=NUM_CPUS)

        ordered_files = [rec.path for rec in block_records.ravel() if rec != 0]
        ordered_files = [output_segy_path] + ordered_files

        if client is not None:
            _ = client.submit(concat_files, paths=ordered_files).result()
        else:
            concat_files(paths=ordered_files, progress=True)


def _zero_fill_variable(src_ds: xr.Dataset, trace_variable_name: str) -> None:
    var_to_zero_fill = src_ds[trace_variable_name]
    if not hasattr(var_to_zero_fill.data, "chunks"):
        err = "The source dataset is not dask-backed."
        raise RuntimeError(err)

    fill_value = var_to_zero_fill.attrs.get("_FillValue", 0)
    # Create a Dask array with the same shape, dtype, and chunks,
    # but filled with a constant value.
    # This operation is lazy and doesn't compute any data yet.
    empty_array = da.full(
        var_to_zero_fill.shape, fill_value=fill_value, dtype=var_to_zero_fill.dtype, chunks=var_to_zero_fill.chunks
    )

    # Replace the 'traceVariableName' DataArray in the dataset with the new dummy array.
    # This happens in memory (for the Dask graph, not the data itself).
    src_ds[trace_variable_name] = (var_to_zero_fill.dims, empty_array, var_to_zero_fill.attrs)


def copy_mdio(  # noqa: PLR0913
    source: StorageLocation,
    destination: StorageLocation,
    with_traces: bool = True,
    with_headers: bool = True,
    overwrite: bool = False,
) -> None:
    """Copy an MDIO dataset to another location.

    Copies an MDIO dataset to another location, optionally setting headers and / or
    traces to the fill values (fill values are not storage-persisted).

    Args:
        source: The source MDIO dataset location.
        destination: The destination MDIO dataset location.
        with_traces: Whether to include traces data in the copy.
        with_headers: Whether to include headers data in the copy.
        overwrite: Whether to overwrite the destination if it exists.

    Raises:
        FileExistsError: If the destination exists and overwrite is False.
    """
    # Open source dataset with dask-backed arrays
    # NOTE: Xarray will convert int to float and replace _FillValue with NaN
    # NOTE: 'chunks={}' instructs xarray to load the dataset into dask arrays using the engine's
    # preferred chunk sizes if they are exposed by the backend. 'chunks="auto"' instructs xarray
    # to use dask's automatic chunking algorithm. This algorithm attempts to determine optimal
    # chunk sizes based on factors like array size and available memory, while also taking
    # into account any engine-preferred chunks if they exist.
    src_ds = xr.open_dataset(source.uri, engine="zarr", chunks="auto", mask_and_scale=False)

    if not overwrite and destination.exists():
        err = f"Output location '{destination.uri}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    if not with_headers:
        _zero_fill_variable(src_ds, "headers")

    if not with_traces:
        # The "traceVariableName" must be in attributes for the later versions
        # HACK: Using a "amplitude" as the default is a temporary workaround since it
        # might not be defined in early versions
        trace_variable_name = src_ds.attrs["attributes"].get("traceVariableName", "amplitude")
        _zero_fill_variable(src_ds, trace_variable_name)

    # Save to destination Zarr store, writing chunk by chunk
    src_ds.to_zarr(destination.uri, mode="w", compute=True, zarr_format=2)


def copy_mdio_cli(  # noqa PLR0913
    source_mdio_path: str,
    target_mdio_path: str,
    overwrite: bool,
    with_traces: bool,
    with_headers: bool,
    storage_options_input: dict | None,
    storage_options_output: dict | None,
) -> None:
    """CLI wrapper for copy_mdio function.

    Args:
        source_mdio_path: Path to the source MDIO dataset location.
        target_mdio_path: Path to the destination MDIO dataset location.
        overwrite: Whether to overwrite the destination if it exists.
        with_traces: Whether to include traces in the copy.
        with_headers: Whether to include headers in the copy.
        storage_options_input: Cloud storage options for the source MDIO dataset.
        storage_options_output: Cloud storage options for the destination MDIO dataset.
    """
    copy_mdio(
        source=StorageLocation(source_mdio_path, storage_options_input),
        destination=StorageLocation(target_mdio_path, storage_options_output),
        with_traces=with_traces,
        with_headers=with_headers,
        overwrite=overwrite,
    )
