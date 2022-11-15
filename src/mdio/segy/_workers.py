"""Low level workers for parsing and writing SEG-Y to Zarr."""


from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
import segyio
from numpy.typing import ArrayLike
from zarr import Array

from mdio.constants import UINT32_MAX
from mdio.core import Grid
from mdio.segy.byte_utils import ByteOrder


def header_scan_worker(
    segy_path_or_handle: str | segyio.SegyFile,
    trace_range: Sequence[int],
    byte_locs: Sequence[int],
    byte_lengths: Sequence[int],
    segy_endian: str,
) -> ArrayLike:
    """Header scan worker.

    Can accept file path or segyio.SegyFile.

    segyio.SegyFile is recommended in case it is called from another process
    that already opened the file (i.e. trace_worker).

    If SegyFile is not open, it can either accept a path string or a handle
    that was opened in a different context manager.

    Args:
        segy_path_or_handle: Path or handle to the input SEG-Y file
        byte_locs: Byte locations to return. It will be a subset of the headers.
        byte_lengths: Tuple consisting of the byte lengths for the index
            attributes. None sets it to 4 per index
        trace_range: Tuple consisting of the trace ranges to read
        segy_endian: Endianness of the input SEG-Y. Rev.2 allows little endian

    Returns:
        Numpy array of parsed headers for the current block.

    Raises:
        TypeError: if segy_path_or_handle is incorrect / unsupported.
    """
    start, stop = trace_range

    if isinstance(segy_path_or_handle, str):
        with segyio.open(
            filename=segy_path_or_handle,
            mode="r",
            ignore_geometry=True,
            endian=segy_endian,
        ) as segy_handle:

            block_headers = [
                segy_handle.header[trc_idx] for trc_idx in range(start, stop)
            ]

    elif isinstance(segy_path_or_handle, segyio.SegyFile):
        block_headers = [
            segy_path_or_handle.header[trc_idx] for trc_idx in range(start, stop)
        ]

    else:
        raise TypeError("Unsupported type for segy_path_or_handle")

    # We keep only the ones we want here (if there is a subset).
    # Sometimes we have custom header locations that are not SEG-Y Std Rev 1.
    # In this case we can't use segyio's byte unpacking anymore.
    # First we create a struct to unpack the 240-byte trace headers.
    # The struct only knows about dimension keys, and their byte offsets.
    # Pads the rest of the data with voids.
    endian = ByteOrder[segy_endian.upper()]
    struct_dtype = np.dtype(
        {
            "names": [f"dim_{idx}" for idx in range(len(byte_locs))],
            "formats": [endian + "i" + str(length) for length in byte_lengths],
            "offsets": [byte_loc - 1 for byte_loc in byte_locs],
            "itemsize": 240,
        }
    )

    # Then for each trace header, we take the unpacked byte buffer from segyio
    # and join them into one byte array. Then we use numpy's frombuffer() to unpack
    block_headers = b"".join([trace_headers.buf for trace_headers in block_headers])
    n_traces = stop - start
    block_headers = np.frombuffer(block_headers, struct_dtype, count=n_traces)
    block_headers = [block_headers[dim] for dim in block_headers.dtype.names]
    return np.column_stack(block_headers)


def trace_worker(
    segy_path: str,
    data_array: Array,
    metadata_array: Array,
    grid: Grid,
    chunk_indices: tuple[slice, ...],
    segy_endian: str,
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
        segy_path: Path to the input SEG-Y file
        data_array: Handle for zarr.Array we are writing traces to
        metadata_array: Handle for zarr.Array we are writing trace headers
        grid: mdio.Grid instance
        chunk_indices: Tuple consisting of the chunk slice indices for
            each dimension
        segy_endian: Endianness of the input SEG-Y. Rev.2 allows little endian

    Returns:
        Partial statistics for chunk, or None

    """
    # Special case where there are no traces inside chunk.
    live_subset = grid.live_mask[chunk_indices[:-1]]
    n_dim = grid.ndim
    if np.count_nonzero(live_subset) == 0:
        return

    # Let's get trace numbers from grid map using the chunk indices.
    seq_trace_indices = grid.map[chunk_indices[:-1]]

    tmp_data = np.zeros(
        seq_trace_indices.shape + (grid.shape[-1],), dtype=data_array.dtype
    )
    tmp_metadata = np.zeros(seq_trace_indices.shape, dtype=metadata_array.dtype)

    del grid  # To save some memory

    # Read headers and traces for block
    with segyio.open(
        filename=segy_path, mode="r", ignore_geometry=True, endian=segy_endian
    ) as segy_handle:
        # Here we utilize ndenumerate so it is dimension agnostic!
        # We don't have to write custom implementations (nested loops) for each case.
        for index, trace_num in np.ndenumerate(seq_trace_indices):
            # We check if the trace is "valid" or "live"
            # 4294967295 is our NULL value. uint32 equivalent of -1. (max uint32)
            if trace_num == UINT32_MAX:
                continue

            # Read header and trace
            # We tested trace vs trace.raw on a single trace
            # They are the same performance. Keeping the lazy version here.
            tmp_metadata[index] = tuple(segy_handle.header[trace_num].values())
            tmp_data[index] = segy_handle.trace[trace_num]

    # Flush metadata to zarr
    metadata_array.set_basic_selection(
        selection=chunk_indices[:-1],
        value=tmp_metadata,
    )

    nonzero_z = tmp_data.sum(axis=tuple(range(n_dim - 1))).nonzero()
    if len(nonzero_z[0]) == 0:
        return

    dimn_start = np.min(nonzero_z)
    dimn_end = np.max(nonzero_z) + 1

    z_slice = slice(dimn_start, dimn_end)
    # We write if there are any values
    chunk_indices = chunk_indices[:-1] + (z_slice,)
    data_array.set_basic_selection(
        selection=chunk_indices,
        value=tmp_data[..., z_slice],
    )

    # Calculate statistics
    nonzero_mask = tmp_data != 0
    count = nonzero_mask.sum(dtype="uint32")

    tmp_data = tmp_data[nonzero_mask]
    chunk_sum = tmp_data.sum(dtype="float64")
    chunk_sum_squares = np.square(tmp_data, dtype="float64").sum()
    min_val = tmp_data.min()
    max_val = tmp_data.max()

    return count, chunk_sum, chunk_sum_squares, min_val, max_val


# tqdm only works properly with pool.map
# However, we need pool.starmap because we have more than one
# argument to make pool.map work with multiple arguments, we
# wrap the function and consolidate arguments to one
def trace_worker_map(args):
    """Wrapper for trace worker to use with tqdm."""
    return trace_worker(*args)


def header_scan_worker_map(args):
    """Wrapper for header scan worker to use with tqdm."""
    return header_scan_worker(*args)
