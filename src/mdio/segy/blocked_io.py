"""Functions for doing blocked I/O from SEG-Y."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import TYPE_CHECKING

import numpy as np
from dask.array import Array
from dask.array import map_blocks
from psutil import cpu_count
from tqdm.auto import tqdm

from mdio.core import Grid
from mdio.core.indexing import ChunkIterator
from mdio.segy._workers import trace_worker
from mdio.segy.creation import SegyPartRecord
from mdio.segy.creation import concat_files
from mdio.segy.creation import serialize_to_segy_stack
from mdio.segy.utilities import find_trailing_ones_index


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from segy import SegyFactory
    from segy import SegyFile


default_cpus = cpu_count(logical=True)


def to_zarr(
    segy_file: SegyFile,
    grid: Grid,
    data_array: Array,
    header_array: Array,
) -> dict:
    """Blocked I/O from SEG-Y to chunked `zarr.core.Array`.

    Args:
        segy_file: SEG-Y file instance.
        grid: mdio.Grid instance
        data_array: Handle for zarr.core.Array we are writing trace data
        header_array: Handle for zarr.core.Array we are writing trace headers

    Returns:
        Global statistics for the SEG-Y as a dictionary.
    """
    # Initialize chunk iterator (returns next chunk slice indices each iteration)
    chunker = ChunkIterator(data_array, chunk_samples=False)
    num_chunks = len(chunker)

    # For Unix async writes with s3fs/fsspec & multiprocessing,
    # use 'spawn' instead of default 'fork' to avoid deadlocks
    # on cloud stores. Slower but necessary. Default on Windows.
    num_cpus = int(os.getenv("MDIO__IMPORT__CPU_COUNT", default_cpus))
    num_workers = min(num_chunks, num_cpus)
    context = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=context)

    # Chunksize here is for multiprocessing, not Zarr chunksize.
    pool_chunksize, extra = divmod(num_chunks, num_workers * 4)
    pool_chunksize += 1 if extra else pool_chunksize

    tqdm_kw = dict(unit="block", dynamic_ncols=True)
    with executor:
        lazy_work = executor.map(
            trace_worker,  # fn
            repeat(segy_file),
            repeat(data_array),
            repeat(header_array),
            repeat(grid),
            chunker,
            chunksize=pool_chunksize,
        )

        lazy_work = tqdm(
            iterable=lazy_work,
            total=num_chunks,
            desc=f"Ingesting SEG-Y in {num_chunks} chunks",
            **tqdm_kw,
        )

        # This executes the lazy work.
        chunk_stats = list(lazy_work)

    # This comes in as n_chunk x 5 columns.
    # Columns in order: count, sum, sum of squared, min, max.
    # From here we can compute global mean, std, rms, min, max.
    # Transposing because we want each statistic as a row to unpack later.
    # REF: https://math.stackexchange.com/questions/1547141/aggregating-standard-deviation-to-a-summary-point  # noqa: B950
    # REF: https://www.mathwords.com/r/root_mean_square.htm
    chunk_stats = [stat for stat in chunk_stats if stat is not None]

    chunk_stats = zip(*chunk_stats)  # noqa: B905
    glob_count, glob_sum, glob_sum_square, glob_min, glob_max = chunk_stats

    glob_count = np.sum(glob_count)  # Comes in as `uint32`
    glob_sum = np.sum(glob_sum)  # `float64`
    glob_sum_square = np.sum(glob_sum_square)  # `float64`
    glob_min = np.min(glob_min)  # `float32`
    glob_max = np.max(glob_max)  # `float32`

    glob_mean = glob_sum / glob_count
    glob_std = np.sqrt(glob_sum_square / glob_count - (glob_sum / glob_count) ** 2)
    glob_rms = np.sqrt(glob_sum_square / glob_count)

    # We need to write these as float64 because float32 is not JSON serializable
    # Trace data is originally float32, hence min/max
    glob_min = glob_min.min().astype("float64")
    glob_max = glob_max.max().astype("float64")

    stats = {
        "mean": glob_mean,
        "std": glob_std,
        "rms": glob_rms,
        "min": glob_min,
        "max": glob_max,
    }

    return stats


def segy_record_concat(
    block_records: NDArray,
    file_root: str,
    block_info: dict | None = None,
) -> NDArray:
    """Concatenate partial ordered SEG-Y blocks on disk.

    It will take an ND array SegyPartRecords. Goal is to preserve
    the global order of traces when merging files. Order is assumed
    to be correct at the block level (last dimension)

    Args:
        block_records: Array indicating block file records.
        file_root: Root directory to write partial SEG-Y files.
        block_info: Dask map_blocks reserved kwarg for block indices / shape etc.

    Returns:
        Concatenated SEG-Y block records.

    Raises:
        ValueError: If required `block_info` is not provided.
    """
    if block_info is None:
        raise ValueError("block_info is required for global index computation.")

    if np.count_nonzero(block_records) == 0:
        return np.zeros_like(block_records, shape=block_records.shape[:-1])

    info = block_info[0]

    block_start = [loc[0] for loc in info["array-location"]]

    record_shape = block_records.shape[:-1]
    records_metadata = np.zeros(shape=record_shape, dtype=object)

    dest_map = {}
    for rec_index in np.ndindex(record_shape):
        rec_blocks = block_records[rec_index]

        if np.count_nonzero(rec_blocks) == 0:
            continue

        global_index = tuple(
            block_start[i] + rec_index[i] for i in range(len(record_shape))
        )
        record_id_str = "/".join(map(str, global_index))
        record_file_path = f"{file_root}/{record_id_str}.bin"

        records_metadata[rec_index] = SegyPartRecord(
            path=record_file_path,
            index=global_index,
        )

        if record_file_path not in dest_map:
            dest_map[record_file_path] = []

        for block in rec_blocks:
            if block == 0:
                continue
            dest_map[record_file_path].append(block.path)

    for dest_path, source_paths in dest_map.items():
        concat_files([dest_path] + source_paths)

    return records_metadata


def to_segy(
    samples: Array,
    headers: Array,
    live_mask: Array,
    segy_factory: SegyFactory,
    file_root: str,
) -> Array:
    r"""Convert MDIO blocks to SEG-Y parts.

    Blocks are written out in parallel via multiple workers, and then
    djacent blocks are tracked and merged on disk via the `segy_trace_concat`
    function. The adjacent are hierarchically merged, and it preserves order.

    Assume array with shape (4, 3, 2) with chunk sizes (1, 1, 2).
    The chunk indices for this array would be:

    (0, 0, 0) (0, 1, 0) (0, 2, 0)
    (1, 0, 0) (1, 1, 0) (1, 2, 0)
    (2, 0, 0) (2, 1, 0) (2, 2, 0)
    (3, 0, 0) (3, 1, 0) (3, 2, 0)

    let's rename them to this for convenience:

    a b c
    d e f
    g h i
    j k l

    The tree gets formed this way:
    a b c d e f g h i
    \/  | \/  | \/  |
    ab  c de  f gh  i
      \/    \/    \/
     abc   def   ghi

    During all the processing here, we keep track of logical indices of
    chunks and written files so we can correctly combine them. The above
    algorithm generalizes to higher dimensions.

    Args:
        samples: Sample array.
        headers: Header array.
        live_mask: Live mask array.
        segy_factory: A SEG-Y factory configured to write out with user params.
        file_root: Root directory to write partial SEG-Y files.

    Returns:
        Array containing live (written) status of final flattened SEG-Y blocks.
    """
    # Calculate axes with only one chunk to be reduced
    num_blocks = samples.numblocks
    non_consecutive_axes = find_trailing_ones_index(num_blocks)
    reduce_axes = tuple(
        i
        for i in range(non_consecutive_axes - 1, len(num_blocks))
        if num_blocks[i] == 1
    )

    # Append headers, and write block as stack of SEG-Ys (full sample dim).
    # Output is N-1 dimensions. We merged headers + samples to new dtype.
    meta = np.empty((1,), dtype=object)
    block_io_records = map_blocks(
        serialize_to_segy_stack,
        samples,
        headers[..., None],  # pad sample dim
        live_mask[..., None],  # pad sample dim
        record_ndim=non_consecutive_axes,
        file_root=file_root,
        segy_factory=segy_factory,
        drop_axis=reduce_axes,
        block_info=True,
        meta=meta,
    )

    # Recursively combine SEG-Y files from last (fastest) consecutive dimension
    # to first (slowest) dimension. End result will be the blocks with the
    # size of the outermost dimension in ascending order.
    while non_consecutive_axes > 1:
        current_chunks = block_io_records.chunks

        prefix_dim = non_consecutive_axes - 1
        prefix_chunks = current_chunks[:prefix_dim]
        new_chunks = prefix_chunks + (-1,) * (len(current_chunks) - prefix_dim)

        block_io_records = map_blocks(
            segy_record_concat,
            block_io_records.rechunk(new_chunks),
            file_root=file_root,
            drop_axis=-1,
            block_info=True,
            meta=meta,
        )

        non_consecutive_axes -= 1

    return block_io_records
