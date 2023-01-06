"""Parsers for sections of SEG-Y files."""


from __future__ import annotations

from itertools import repeat
from math import ceil
from multiprocessing import Pool
from typing import Any
from typing import Sequence

import numpy as np
import segyio
from psutil import cpu_count
from tqdm.auto import tqdm

from mdio.core import Dimension
from mdio.segy._workers import header_scan_worker_map


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
    byte_lengths: Sequence[int],
    block_size: int = 50000,
    progress_bar: bool = True,
) -> np.ndarray:
    """Read and parse given `byte_locations` from SEG-Y file.

    Args:
        segy_path: Path to the input SEG-Y file.
        segy_endian: Endianness of the input SEG-Y in {'big', 'little'}.
        byte_locs: Byte locations to return. It will be a subset of headers.
        byte_lengths: Byte lengths of each header key to index. Must be the
            same count as `byte_lengths`.
        block_size: Number of traces to read for each block.
        progress_bar: Enable or disable progress bar. Default is True.

    Returns:
        Numpy array of parsed trace headers.
    """
    trace_count = get_trace_count(segy_path, segy_endian)
    n_blocks = int(ceil(trace_count / block_size))

    trace_ranges = []
    for idx in range(n_blocks):
        start, stop = idx * block_size, (idx + 1) * block_size
        if stop > trace_count:
            stop = trace_count

        trace_ranges.append((start, stop))

    # Note: Make sure the order of this is exactly
    # the same as the function call.
    parallel_inputs = zip(  # noqa: B905 or strict=False >= py3.10
        repeat(segy_path),
        trace_ranges,
        repeat(byte_locs),
        repeat(byte_lengths),
        repeat(segy_endian),
    )

    num_workers = min(n_blocks, NUM_CORES)

    tqdm_kw = dict(unit="block", dynamic_ncols=True)
    with Pool(num_workers) as pool:
        # pool.imap is lazy
        lazy_work = pool.imap(
            func=header_scan_worker_map,
            iterable=parallel_inputs,
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

    # Merge blocks before return
    return np.vstack(headers)


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
