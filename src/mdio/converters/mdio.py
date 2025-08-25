"""Conversion from to MDIO various other formats."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import xarray as xr
from psutil import cpu_count
from tqdm.dask import TqdmCallback

from mdio.segy.blocked_io import to_segy
from mdio.segy.creation import concat_files
from mdio.segy.creation import mdio_spec_to_segy
from mdio.segy.utilities import segy_export_rechunker

try:
    import distributed
except ImportError:
    distributed = None

if TYPE_CHECKING:
    from segy.schema import SegySpec

    from mdio.core.storage_location import StorageLocation

default_cpus = cpu_count(logical=True)
NUM_CPUS = int(os.getenv("MDIO__EXPORT__CPU_COUNT", default_cpus))


def _get_dask_array(mdio_xr: xr.Dataset, var_name: str, chunks: tuple[int, ...] = None) -> da.Array:
    """Workaround if the MDIO Xarray dataset returns numpy arrays instead of Dask arrays"""
    xr_var = mdio_xr[var_name]
    # xr_var.chunks:
    # Tuple of block lengths for this dataarray’s data, in order of dimensions,
    # or None if the underlying data is not a dask array.
    if xr_var.chunks is not None:
        return xr_var.data.rechunk(chunks)
    # For some reason, a NumPy in-memory array was returned
    # HACK: Convert NumPy array to a chunked Dask array
    return da.from_array(xr_var.data, chunks=chunks)


def mdio_to_segy(  # noqa: PLR0912, PLR0913, PLR0915
    segy_spec: SegySpec,
    input_location: StorageLocation,
    output_location: StorageLocation,
    selection_mask: np.ndarray = None,
    client: distributed.Client = None,
) -> None:
    """Convert MDIO file to SEG-Y format.

    We export N-D seismic data to the flattened SEG-Y format used in data transmission.

    The input headers are preserved as is, and will be transferred to the output file.

    Input MDIO can be local or cloud based. However, the output SEG-Y will be generated locally.

    A `selection_mask` can be provided (same shape as spatial grid) to export a subset.

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        input_location: Store or URL (and cloud options) for MDIO file.
        output_location: Path to the output SEG-Y file.
        selection_mask: Array that lists the subset of traces
        client: Dask client. If `None` we will use local threaded scheduler. If `auto` is used we
            will create multiple processes (with 8 threads each).

    Raises:
        ImportError: if distributed package isn't installed but requested.
        ValueError: if cut mask is empty, i.e. no traces will be written.

    Examples:
        To export an existing local MDIO file to SEG-Y we use the code snippet below. This will
        export the full MDIO (without padding) to SEG-Y format.

        >>> from mdio import mdio_to_segy
        >>>
        >>>
        >>> mdio_to_segy(
        ...     mdio_path_or_buffer="prefix2/file.mdio",
        ...     output_segy_path="prefix/file.segy",
        ... )
    """
    output_segy_path = Path(output_location.uri)

    mdio_xr = xr.open_dataset(input_location.uri, engine="zarr", mask_and_scale=False)

    trace_variable_name = mdio_xr.attrs["attributes"]["traceVariableName"]
    amplitude = mdio_xr[trace_variable_name]
    chunks: tuple[int, ...] = amplitude.encoding.get("chunks")
    shape: tuple[int, ...] = amplitude.shape
    dtype = amplitude.dtype
    new_chunks = segy_export_rechunker(chunks, shape, dtype)

    creation_args = [segy_spec, input_location, output_location]

    if client is not None:
        if distributed is not None:
            # This is in case we work with big data
            feature = client.submit(mdio_spec_to_segy, *creation_args)
            mdio_xr, segy_factory = feature.result()
        else:
            msg = "Distributed client was provided, but `distributed` is not installed"
            raise ImportError(msg)
    else:
        mdio_xr, segy_factory = mdio_spec_to_segy(*creation_args)

    # Using XArray.DataArray.values should trigger compute and load the whole array into memory.
    live_mask = mdio_xr["trace_mask"].values
    # live_mask = mdio.live_mask.compute()

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
    # Workaround: currently the MDIO Xarray dataset returns numpy arrays instead of Dask arrays
    # TODO (Dmitriy Repin): Revisit after the eager memory allocation is fixed
    # https://github.com/TGSAI/mdio-python/issues/608
    live_mask = _get_dask_array(mdio_xr, "trace_mask", new_chunks[:-1])[dim_slices]
    headers = _get_dask_array(mdio_xr, "headers", new_chunks[:-1])[dim_slices]
    samples = _get_dask_array(mdio_xr, "amplitude", new_chunks)[dim_slices]

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
