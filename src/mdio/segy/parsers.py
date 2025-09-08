"""Parsers for sections of SEG-Y files."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
from psutil import cpu_count
from tqdm.auto import tqdm

from mdio.segy._workers import header_scan_worker

if TYPE_CHECKING:
    from segy import SegyFile
    from segy.arrays import HeaderArray

default_cpus = cpu_count(logical=True)


def parse_headers(
    segy_file: SegyFile,
    subset: list[str] | None = None,
    block_size: int = 10000,
    progress_bar: bool = True,
) -> HeaderArray:
    """Read and parse given `byte_locations` from SEG-Y file.

    Args:
        segy_file: SegyFile instance.
        subset: List of header names to filter and keep.
        block_size: Number of traces to read for each block.
        progress_bar: Enable or disable progress bar. Default is True.

    Returns:
        HeaderArray. Keys are the index names, values are numpy arrays of parsed headers for the
        current block. Array is of type byte_type except IBM32 which is mapped to FLOAT32.
    """
    trace_count = segy_file.num_traces
    n_blocks = int(ceil(trace_count / block_size))

    trace_ranges = []
    for idx in range(n_blocks):
        start, stop = idx * block_size, (idx + 1) * block_size
        stop = min(stop, trace_count)

        trace_ranges.append((start, stop))

    num_cpus = int(os.getenv("MDIO__IMPORT__CPU_COUNT", default_cpus))
    num_workers = min(n_blocks, num_cpus)

    segy_kw = {
        "url": segy_file.fs.unstrip_protocol(segy_file.url),
        "spec": segy_file.spec,
        "settings": segy_file.settings,
    }
    tqdm_kw = {"unit": "block", "dynamic_ncols": True}
    with ProcessPoolExecutor(num_workers) as executor:
        lazy_work = executor.map(header_scan_worker, repeat(segy_kw), trace_ranges, repeat(subset))

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
