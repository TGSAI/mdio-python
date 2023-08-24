"""Parsers for sections of SEG-Y files."""


from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from math import ceil
from typing import Any
from typing import Sequence

import numpy as np
import segyio
from numpy.typing import NDArray
from psutil import cpu_count
from tqdm.auto import tqdm

from mdio.core import Dimension
from mdio.segy._workers import header_scan_worker
from mdio.segy.byte_utils import Dtype


NUM_CORES = cpu_count(logical=False)


def get_trace_count(segy_path, segy_endian):
    """Get trace count from SEG-Y file (size)."""
    with segyio.open(
        filename=segy_path,
        mode="r",
        ignore_geometry=True,
        endian=segy_endian,
    ) as segy_handle:
        trace_count = segy_handle.tracecount

    return trace_count


def parse_binary_header(segy_handle: segyio.SegyFile) -> dict[str, Any]:
    """Parse `segyio.BinField` as python `dict`.

    The `segyio` library returns the binary header as a `BinField` instance,
    which is not serializable. Here we parse it as a dictionary, so we can
    later convert it to JSON.

    Args:
        segy_handle: The SegyFile instance in context.

    Returns:
        Parsed binary header key, value pairs.
    """
    binary_header = segy_handle.bin
    return {str(entry): binary_header[entry] for entry in binary_header}


def parse_text_header(segy_handle: segyio.SegyFile) -> list[str]:
    """Parse text header from `bytearray` to python `list` of `str` per line.

    The `segyio` library returns the text header as a  `bytearray` instance.
     Here we parse it as a list of strings (per line).

    Args:
        segy_handle: The SegyFile instance in context.

    Returns:
        Parsed text header in list with lines as elements.
    """
    text_header = segy_handle.text[0].decode(errors="ignore")
    text_header = [
        text_header[char_idx : char_idx + 80]
        for char_idx in range(0, len(text_header), 80)
    ]
    return text_header


def parse_trace_headers(
    segy_path: str,
    segy_endian: str,
    byte_locs: Sequence[int],
    byte_types: Sequence[Dtype],
    index_names: Sequence[str],
    block_size: int = 50000,
    progress_bar: bool = True,
) -> dict[str, NDArray]:
    """Read and parse given `byte_locations` from SEG-Y file.

    Args:
        segy_path: Path to the input SEG-Y file.
        segy_endian: Endianness of the input SEG-Y in {'big', 'little'}.
        byte_locs: Byte locations to return. It will be a subset of headers.
        byte_types: Data types of each header key to index. Must be the
            same count as `byte_lengths`.
        index_names: Tuple of the names for the index attributes
        block_size: Number of traces to read for each block.
        progress_bar: Enable or disable progress bar. Default is True.

    Returns:
        dictionary with headers:  keys are the index names, values are numpy
            arrays of parsed headers for the current block. Array is of type
            byte_type with the exception of IBM32 which is mapped to FLOAT32.

    """
    trace_count = get_trace_count(segy_path, segy_endian)
    n_blocks = int(ceil(trace_count / block_size))

    trace_ranges = []
    for idx in range(n_blocks):
        start, stop = idx * block_size, (idx + 1) * block_size
        if stop > trace_count:
            stop = trace_count

        trace_ranges.append((start, stop))

    num_workers = min(n_blocks, NUM_CORES)

    tqdm_kw = dict(unit="block", dynamic_ncols=True)
    with ProcessPoolExecutor(num_workers) as executor:
        # pool.imap is lazy
        lazy_work = executor.map(
            header_scan_worker,  # fn
            repeat(segy_path),
            trace_ranges,
            repeat(byte_locs),
            repeat(byte_types),
            repeat(index_names),
            repeat(segy_endian),
            chunksize=2,  # Not array chunks. This is for `multiprocessing`
        )

        if progress_bar is True:
            lazy_work = tqdm(
                iterable=lazy_work,
                total=n_blocks,
                desc="Scanning SEG-Y for geometry attributes",
                **tqdm_kw,
            )

        # This executes the lazy work.
        headers = list(lazy_work)

    final_headers = {}
    for header_name in index_names:
        final_headers[header_name] = np.concatenate(
            [header[header_name] for header in headers]
        )
    # Merge blocks before return
    return final_headers


def parse_sample_axis(binary_header: dict) -> Dimension:
    """Parse sample axis from binary header.

    Args:
        binary_header: Dictionary representing binary header.

    Returns:
        MDIO Dimension instance for sample axis.
    """
    num_samp = binary_header["Samples"]
    interval = binary_header["Interval"] // 1000

    max_val = num_samp * interval

    return Dimension(coords=range(0, max_val, interval), name="sample")
