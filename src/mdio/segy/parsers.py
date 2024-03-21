"""Parsers for sections of SEG-Y files."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
from psutil import cpu_count
from tqdm.auto import tqdm

from mdio.segy._workers import header_scan_worker


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from segy import SegyFile

NUM_CORES = cpu_count(logical=True)


def parse_trace_headers(
    segy_file: SegyFile,
    index_names: list[str],
    block_size: int = 50000,
    progress_bar: bool = True,
) -> NDArray:
    """Read and parse given `byte_locations` from SEG-Y file.

    Args:
        segy_file: SegyFile instance.
        index_names: Tuple of the names for the index attributes.
        block_size: Number of traces to read for each block.
        progress_bar: Enable or disable progress bar. Default is True.

    Returns:
        dictionary with headers:  keys are the index names, values are numpy
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

    num_workers = min(n_blocks, NUM_CORES)

    tqdm_kw = dict(unit="block", dynamic_ncols=True)
    with ProcessPoolExecutor(num_workers) as executor:
        # pool.imap is lazy
        lazy_work = executor.map(
            header_scan_worker,  # fn
            repeat(segy_file),
            trace_ranges,
            repeat(index_names),
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
    return np.concatenate(headers)
