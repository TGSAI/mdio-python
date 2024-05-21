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
from mdio.segy.byte_utils import Dtype
from mdio.segy.ibm_float import ibm2ieee


def header_scan_worker(
    segy_path_or_handle: str | segyio.SegyFile,
    trace_range: Sequence[int],
    byte_locs: Sequence[int],
    byte_types: Sequence[Dtype],
    index_names: Sequence[str],
    segy_endian: str,
) -> dict[str, ArrayLike]:
    """Header scan worker.

    Can accept file path or segyio.SegyFile.

    segyio.SegyFile is recommended in case it is called from another process
    that already opened the file (i.e. trace_worker).

    If SegyFile is not open, it can either accept a path string or a handle
    that was opened in a different context manager.

    Args:
        segy_path_or_handle: Path or handle to the input SEG-Y file
        byte_locs: Byte locations to return. It will be a subset of the headers.
        byte_types: Tuple consisting of the data types for the index attributes.
        trace_range: Tuple consisting of the trace ranges to read
        index_names: Tuple of the names for the index attributes
        segy_endian: Endianness of the input SEG-Y. Rev.2 allows little endian

    Returns:
        dictionary with headers:  keys are the index names, values are numpy
            arrays of parsed headers for the current block. Array is of type
            byte_type with the exception of IBM32 which is mapped to FLOAT32.

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
    # NOTE: segyio buffer is always big endian. This is why we force it here.
    # This used to be the same as `segy_endian` but we hard code it to big.
    endian = ByteOrder.BIG

    # Handle byte offsets
    offsets = [0 if byte_loc is None else byte_loc - 1 for byte_loc in byte_locs]
    formats = [type_.numpy_dtype.newbyteorder(endian) for type_ in byte_types]

    struct_dtype = np.dtype(
        {
            "names": index_names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": 240,
        }
    )

    # Then for each trace header, we take the unpacked byte buffer from segyio
    # and join them into one byte array. Then we use numpy's frombuffer() to unpack
    block_headers = b"".join([trace_headers.buf for trace_headers in block_headers])
    n_traces = stop - start
    block_headers = np.frombuffer(block_headers, struct_dtype, count=n_traces)
    block_headers = {name: block_headers[name] for name in index_names}

    out_dtype = []
    for name, type_ in zip(index_names, byte_types):  # noqa: B905
        if type_ == Dtype.IBM32:
            native_dtype = Dtype.FLOAT32.numpy_dtype
        else:
            native_dtype = type_.numpy_dtype

        out_dtype.append((name, native_dtype))

    # out_array = np.empty(n_traces, out_dtype)
    out_array = {}

    # TODO: Add strict=True and remove noqa when minimum Python is 3.10
    for name, loc, type_ in zip(index_names, byte_locs, byte_types):  # noqa: B905
        # Handle exception when a byte_loc is None
        if loc is None:
            out_array[name] = 0
            del block_headers[name]
            continue

        header = block_headers[name]

        if type_ == Dtype.IBM32:
            header = ibm2ieee(header)

        out_array[name] = header

        del block_headers[name]

    return out_array


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

    # Find first non-zero index in sample (last) dimension
    non_sample_axes = tuple(range(n_dim - 1))
    nonzero_z = np.where(np.any(tmp_data != 0, axis=non_sample_axes))

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
