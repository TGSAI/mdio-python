"""Parsers for sections of SEG-Y files."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
from psutil import cpu_count
from segy.arrays import HeaderArray
from tqdm.auto import tqdm

from mdio.segy._workers import header_scan_worker


if TYPE_CHECKING:
    from segy import SegyFile

default_cpus = cpu_count(logical=True)
NUM_CPUS = int(os.getenv("MDIO__IMPORT__CPU_COUNT", default_cpus))


def parse_index_headers(
    segy_file: SegyFile,
    block_size: int = 10000,
    progress_bar: bool = True,
) -> HeaderArray:
    """Read and parse given `byte_locations` from SEG-Y file.

    Args:
        segy_file: SegyFile instance.
        block_size: Number of traces to read for each block.
        progress_bar: Enable or disable progress bar. Default is True.

    Returns:
        Numpy array of headers. Keys are the index names, values are numpy
            arrays of parsed headers for the current block. Array is of type
            byte_type except IBM32 which is mapped to FLOAT32.
    """
    trace_count = segy_file.num_traces
    n_blocks = int(ceil(trace_count / block_size))

    trace_ranges = []
    for idx in range(n_blocks):
        start, stop = idx * block_size, (idx + 1) * block_size
        if stop > trace_count:
            stop = trace_count

        trace_ranges.append((start, stop))

    num_workers = min(n_blocks, NUM_CPUS)

    tqdm_kw = dict(unit="block", dynamic_ncols=True)
    with ProcessPoolExecutor(num_workers) as executor:
        # pool.imap is lazy
        lazy_work = executor.map(
            header_scan_worker,  # fn
            repeat(segy_file),
            trace_ranges,
        )

        if progress_bar is True:
            lazy_work = tqdm(
                iterable=lazy_work,
                total=n_blocks,
                desc="Scanning SEG-Y for geometry attributes",
                **tqdm_kw,
            )

        # This executes the lazy work.
        headers: list[HeaderArray] = list(lazy_work)

    # Merge blocks before return
    return np.concatenate(headers)
