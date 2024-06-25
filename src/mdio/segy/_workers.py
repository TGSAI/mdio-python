"""Low level workers for parsing and writing SEG-Y to Zarr."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np


if TYPE_CHECKING:
    from segy import SegyFile
    from segy.arrays import HeaderArray
    from zarr import Array

    from mdio.core import Grid


def header_scan_worker(
    segy_file: SegyFile,
    trace_range: tuple[int, int],
) -> HeaderArray:
    """Header scan worker.

    Can accept file path or segyio.SegyFile.

    segyio.SegyFile is recommended in case it is called from another process
    that already opened the file (i.e. trace_worker).

    If SegyFile is not open, it can either accept a path string or a handle
    that was opened in a different context manager.

    Args:
        segy_file: SegyFile instance.
        trace_range: Tuple consisting of the trace ranges to read.

    Returns:
        HeaderArray parsed from SEG-Y library.
    """
    return segy_file.header[slice(*trace_range)]


def trace_worker(
    segy_file: SegyFile,
    data_array: Array,
    metadata_array: Array,
    grid: Grid,
    chunk_indices: tuple[slice, ...],
) -> tuple[Any, ...] | None:
    """Worker function for multi-process enabled blocked SEG-Y I/O.

    Performance of `zarr.Array` writes are very slow if data being written is
    not aligned with the chunk boundaries. Because of this, we sacrifice
    sequential reads of SEG-Y files. However, won't be an issue if we have
    SSDs or are on cloud.

    It takes the trace numbers from grid and gets the current chunk's trace
    indices (on SEG-Y). Then we fill a temporary array in memory and do a
    write to the `zarr.Array` chunk. In this case we take full slices across
    sample dimension because SEG-Y data is not chunked, so we don't have to
    worry about it.

    Args:
        segy_file: SegyFile instance.
        data_array: Handle for zarr.Array we are writing traces to
        metadata_array: Handle for zarr.Array we are writing trace headers
        grid: mdio.Grid instance
        chunk_indices: Tuple consisting of the chunk slice indices for
            each dimension

    Returns:
        Partial statistics for chunk, or None
    """
    # Special case where there are no traces inside chunk.
    live_subset = grid.live_mask[chunk_indices[:-1]]

    if np.count_nonzero(live_subset) == 0:
        return None

    # Let's get trace numbers from grid map using the chunk indices.
    seq_trace_indices = grid.map[chunk_indices[:-1]]

    tmp_data = np.zeros(
        seq_trace_indices.shape + (grid.shape[-1],), dtype=data_array.dtype
    )

    tmp_metadata = np.zeros(seq_trace_indices.shape, dtype=metadata_array.dtype)

    del grid  # To save some memory

    # Read headers and traces for block
    valid_indices = seq_trace_indices[live_subset]

    traces = segy_file.trace[valid_indices.tolist()]
    headers, samples = traces["header"], traces["data"]

    tmp_metadata[live_subset] = headers.view(tmp_metadata.dtype)
    tmp_data[live_subset] = samples

    # Flush metadata to zarr
    metadata_array.set_basic_selection(
        selection=chunk_indices[:-1],
        value=tmp_metadata,
    )

    nonzero_mask = samples != 0
    nonzero_count = nonzero_mask.sum(dtype="uint32")

    if nonzero_count == 0:
        return None

    data_array.set_basic_selection(
        selection=chunk_indices,
        value=tmp_data,
    )

    # Calculate statistics
    tmp_data = samples[nonzero_mask]
    chunk_sum = tmp_data.sum(dtype="float64")
    chunk_sum_squares = np.square(tmp_data, dtype="float64").sum()
    min_val = tmp_data.min()
    max_val = tmp_data.max()

    return nonzero_count, chunk_sum, chunk_sum_squares, min_val, max_val
