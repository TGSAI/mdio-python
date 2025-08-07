"""Low level workers for parsing and writing SEG-Y to Zarr."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Any
from typing import TypedDict
from typing import cast

import numpy as np
from segy import SegyFile

if TYPE_CHECKING:
    from segy.arrays import HeaderArray
    from segy.config import SegySettings
    from segy.schema import SegySpec
    from zarr import Array

    from mdio.core import Grid


class SegyFileArguments(TypedDict):
    """Arguments to open SegyFile instance creation."""

    url: str
    spec: SegySpec | None
    settings: SegySettings | None


def header_scan_worker(
    segy_kw: SegyFileArguments,
    trace_range: tuple[int, int],
) -> HeaderArray:
    """Header scan worker.

    If SegyFile is not open, it can either accept a path string or a handle that was opened in
    a different context manager.

    Args:
        segy_kw: Arguments to open SegyFile instance.
        trace_range: Tuple consisting of the trace ranges to read.

    Returns:
        HeaderArray parsed from SEG-Y library.
    """
    segy_file = SegyFile(**segy_kw)

    slice_ = slice(*trace_range)

    cloud_native_mode = os.getenv("MDIO__IMPORT__CLOUD_NATIVE", default="False")

    if cloud_native_mode.lower() in {"true", "1"}:
        trace_header = segy_file.trace[slice_].header
    else:
        trace_header = segy_file.header[slice_]

    # Get non-void fields from dtype and copy to new array for memory efficiency
    fields = trace_header.dtype.fields
    non_void_fields = [(name, dtype) for name, (dtype, _) in fields.items()]
    new_dtype = np.dtype(non_void_fields)

    # Copy to non-padded memory, ndmin is to handle the case where there is 1 trace in block
    # (singleton) so we can concat and assign stuff later.
    trace_header = np.array(trace_header, dtype=new_dtype, ndmin=1)

    return cast("HeaderArray", trace_header)


def trace_worker(
    segy_kw: SegyFileArguments,
    data_array: Array,
    metadata_array: Array,
    grid: Grid,
    chunk_indices: tuple[slice, ...],
) -> tuple[Any, ...] | None:
    """Worker function for multi-process enabled blocked SEG-Y I/O.

    Performance of `zarr.Array` writes is slow if data isn't aligned with chunk boundaries,
    sacrificing sequential reads of SEG-Y files. This won't be an issue with SSDs or cloud.

    It retrieves trace numbers from the grid and gathers the current chunk's SEG-Y trace indices.
    Then, it fills a temporary array in memory and writes to the `zarr.Array` chunk. We take full
    slices across the sample dimension since SEG-Y data isn't chunked, eliminating concern.

    Args:
        segy_kw: Arguments to open SegyFile instance.
        data_array: Handle for zarr.Array we are writing traces to
        metadata_array: Handle for zarr.Array we are writing trace headers
        grid: mdio.Grid instance
        chunk_indices: Tuple consisting of the chunk slice indices for each dimension

    Returns:
        Partial statistics for chunk, or None
    """
    # Special case where there are no traces inside chunk.
    segy_file = SegyFile(**segy_kw)
    live_subset = grid.live_mask[chunk_indices[:-1]]

    if np.count_nonzero(live_subset) == 0:
        return None

    # Let's get trace numbers from grid map using the chunk indices.
    seq_trace_indices = grid.map[chunk_indices[:-1]]

    tmp_data = np.zeros(seq_trace_indices.shape + (grid.shape[-1],), dtype=data_array.dtype)
    tmp_metadata = np.zeros(seq_trace_indices.shape, dtype=metadata_array.dtype)

    del grid  # To save some memory

    # Read headers and traces for block
    valid_indices = seq_trace_indices[live_subset]

    traces = segy_file.trace[valid_indices.tolist()]
    headers, samples = traces["header"], traces["data"]

    tmp_metadata[live_subset] = headers.view(tmp_metadata.dtype)
    tmp_data[live_subset] = samples

    # Flush metadata to zarr
    metadata_array.set_basic_selection(selection=chunk_indices[:-1], value=tmp_metadata)

    nonzero_mask = samples != 0
    nonzero_count = nonzero_mask.sum(dtype="uint32")

    if nonzero_count == 0:
        return None

    data_array.set_basic_selection(selection=chunk_indices, value=tmp_data)

    # Calculate statistics
    tmp_data = samples[nonzero_mask]
    chunk_sum = tmp_data.sum(dtype="float64")
    chunk_sum_squares = np.square(tmp_data, dtype="float64").sum()
    min_val = tmp_data.min()
    max_val = tmp_data.max()

    return nonzero_count, chunk_sum, chunk_sum_squares, min_val, max_val
