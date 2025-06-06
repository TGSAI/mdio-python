"""Low level workers for parsing and writing SEG-Y to Zarr."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import numpy as np

if TYPE_CHECKING:
    from segy import SegyFile
    from segy.arrays import HeaderArray
    from zarr import Array

    from mdio.core import Grid


def header_scan_worker(segy_file: SegyFile, trace_range: tuple[int, int]) -> HeaderArray:
    """Header scan worker.

    If SegyFile is not open, it can either accept a path string or a handle that was opened in
    a different context manager.

    Args:
        segy_file: SegyFile instance.
        trace_range: Tuple consisting of the trace ranges to read.

    Returns:
        HeaderArray parsed from SEG-Y library.
    """
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
    segy_file: SegyFile,
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
        segy_file: SegyFile instance.
        data_array: Handle for zarr.Array we are writing traces to
        metadata_array: Handle for zarr.Array we are writing trace headers
        grid: mdio.Grid instance
        chunk_indices: Tuple consisting of the chunk slice indices for each dimension

    Returns:
        Partial statistics for chunk, or None
    """
    # Determine which trace IDs fall into this chunk
    trace_ids = grid.get_traces_for_chunk(chunk_indices[:-1])
    if trace_ids.size == 0:
        return None

    # Read headers and traces for the selected trace IDs
    traces = segy_file.trace[trace_ids.tolist()]
    headers, samples = traces["header"], traces["data"]

    # Build a temporary buffer for data and metadata for this chunk
    chunk_shape = tuple(sli.stop - sli.start for sli in chunk_indices[:-1]) + (grid.shape[-1],)
    tmp_data = np.zeros(chunk_shape, dtype=data_array.dtype)
    meta_shape = tuple(sli.stop - sli.start for sli in chunk_indices[:-1])
    tmp_metadata = np.zeros(meta_shape, dtype=metadata_array.dtype)

    # Compute local coordinates within the chunk for each trace
    local_coords: list[np.ndarray] = []
    for dim_idx, sl in enumerate(chunk_indices[:-1]):
        hdr_arr = grid.header_index_arrays[dim_idx]
        # Optimize memory usage: hdr_arr and trace_ids are already uint32,
        # sl.start is int, so result should naturally be int32/uint32.
        # Avoid unnecessary astype conversion to int64.
        indexed_coords = hdr_arr[trace_ids]  # uint32 array
        local_idx = indexed_coords - sl.start  # remains uint32
        # Only convert dtype if necessary for indexing (numpy requires int for indexing)
        if local_idx.dtype != np.intp:
            local_idx = local_idx.astype(np.intp)
        local_coords.append(local_idx)
    full_idx = tuple(local_coords) + (slice(None),)

    # Populate the temporary buffers
    tmp_data[full_idx] = samples
    tmp_metadata[tuple(local_coords)] = headers.view(tmp_metadata.dtype)

    # Flush metadata to Zarr
    metadata_array.set_basic_selection(selection=chunk_indices[:-1], value=tmp_metadata)

    # Determine nonzero samples and early-exit if none
    nonzero_mask = samples != 0
    nonzero_count = int(nonzero_mask.sum())
    if nonzero_count == 0:
        return None

    # Flush data to Zarr
    data_array.set_basic_selection(selection=chunk_indices, value=tmp_data)

    # Calculate statistics
    flattened_nonzero = samples[nonzero_mask]
    chunk_sum = float(flattened_nonzero.sum(dtype="float64"))
    chunk_sum_squares = float(np.square(flattened_nonzero, dtype="float64").sum())
    min_val = float(flattened_nonzero.min())
    max_val = float(flattened_nonzero.max())

    return (nonzero_count, chunk_sum, chunk_sum_squares, min_val, max_val)
