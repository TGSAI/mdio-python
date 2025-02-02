"""Functions for doing blocked I/O from SEG-Y."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from shutil import copyfileobj
from typing import TYPE_CHECKING

import numpy as np
from dask.array import Array
from dask.array import map_blocks
from psutil import cpu_count
from tqdm.auto import tqdm
from zarr import Blosc
from zarr import Group

from mdio.core import Grid
from mdio.core.indexing import ChunkIterator
from mdio.segy._workers import trace_worker
from mdio.segy.creation import serialize_to_segy_stack
from mdio.segy.utilities import find_trailing_ones_index
from mdio.segy.utilities import ndrange


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from segy import SegyFactory
    from segy import SegyFile

try:
    import zfpy  # Base library
    from zarr import ZFPY  # Codec

except ImportError:
    ZFPY = None
    zfpy = None

default_cpus = cpu_count(logical=True)


def to_zarr(
    segy_file: SegyFile,
    grid: Grid,
    data_root: Group,
    metadata_root: Group,
    name: str,
    chunks: tuple[int, ...],
    lossless: bool,
    compression_tolerance: float = 0.01,
    **kwargs,
) -> dict:
    """Blocked I/O from SEG-Y to chunked `zarr.core.Array`.

    Args:
        segy_file: SEG-Y file instance.
        grid: mdio.Grid instance
        data_root: Handle for zarr.core.Group we are writing traces
        metadata_root: Handle for zarr.core.Group we are writing trace headers
        name: Name of the zarr.Array
        chunks: Chunk sizes for trace data
        lossless: Lossless Blosc with zstandard, or ZFP with fixed precision.
        compression_tolerance: Tolerance ZFP compression, optional. The fixed
            accuracy mode in ZFP guarantees there won't be any errors larger
            than this value. The default is 0.01, which gives about 70%
            reduction in size.
        **kwargs: Additional keyword arguments passed to zarr.core.Array  # noqa: RST210

    Returns:
        Global statistics for the SEG-Y as a dictionary.

    Raises:
        ImportError: if the ZFP isn't installed and user requests lossy.
    """
    if lossless is True:
        trace_compressor = Blosc("zstd")
        header_compressor = trace_compressor
    elif ZFPY is not None or zfpy is not None:
        trace_compressor = ZFPY(
            mode=zfpy.mode_fixed_accuracy,
            tolerance=compression_tolerance,
        )
        header_compressor = Blosc("zstd")
    else:
        raise ImportError(
            "Lossy compression requires the 'zfpy' library. It is "
            "not installed in your environment. To proceed please "
            "install 'zfpy' or install mdio with `--extras lossy`"
        )

    trace_array = data_root.create_dataset(
        name=name,
        shape=grid.shape,
        compressor=trace_compressor,
        chunks=chunks,
        dimension_separator="/",
        write_empty_chunks=False,
        **kwargs,
    )

    # Get header dtype in native order (little-endian 99.9% of the time)
    header_dtype = segy_file.spec.trace.header.dtype.newbyteorder("=")
    header_array = metadata_root.create_dataset(
        name="_".join([name, "trace_headers"]),
        shape=grid.shape[:-1],  # Same spatial shape as data
        chunks=chunks[:-1],  # Same spatial chunks as data
        compressor=header_compressor,
        dtype=header_dtype,
        dimension_separator="/",
        write_empty_chunks=False,
    )

    # Initialize chunk iterator (returns next chunk slice indices each iteration)
    chunker = ChunkIterator(trace_array, chunk_samples=False)
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
            repeat(trace_array),
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


def segy_trace_concat(
    is_block_live: NDArray,
    consecutive_dim_index: int,
    filename_prefix: str,
    block_info: dict | None = None,
) -> NDArray:
    """Concatenate partial ordered SEG-Y blocks on disk.

    It will take an ND array of booleans that indicate if files for a specific
    block exists based on the logical (i, j, ...) coordinates. Goal is to preserve
    the global order of traces when merging files. Order is assumed to be correct
    past consecutive dimension index parameter.

    Args:
        is_block_live: Array indicating block has live traces or not.
        consecutive_dim_index: Dimension to assume ordered files to combine.
        filename_prefix: Prefix directory where files are located.
        block_info: Dask map_blocks reserved kwarg for block indices / shape etc.

    Returns:
        Concatenated live block indicator dropping dimensions past consecutive index.
    """
    if block_info is None:
        return is_block_live.any(axis=-1)

    if np.count_nonzero(is_block_live) == 0:
        return is_block_live.any(axis=-1)

    result_chunk_shape = block_info[None]["chunk-shape"]
    block_coords = block_info[0]["array-location"]

    prefix_block_coords = block_coords[:consecutive_dim_index]
    prefix_block_shape = result_chunk_shape[:consecutive_dim_index]

    # Generate iterators for dimension's index and coords
    indices_iter = np.ndindex(prefix_block_shape)
    coords_iter = ndrange(prefix_block_coords)

    for dim_indices, dim_coords in zip(indices_iter, coords_iter, strict=True):
        aligned_live = is_block_live[*dim_indices]

        if np.count_nonzero(aligned_live) == 0:
            continue

        source_file_index = ".".join(map(str, dim_coords))
        source_path = f"{filename_prefix}/{source_file_index}._mdiotemp"

        dest_file_index = ".".join(map(str, dim_coords[:-1]))
        dest_path = f"{filename_prefix}/{dest_file_index}._mdiotemp"

        with open(dest_path, "ab") as dest, open(source_path, "rb") as src:
            copyfileobj(src, dest)

        os.remove(source_path)

    return is_block_live.any(axis=-1)


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
    # Append headers, and write block as stack of SEG-Ys (full sample dim).
    # Output is N-1 dimensions. We merged headers + samples to new dtype.
    is_block_live = map_blocks(
        serialize_to_segy_stack,
        samples,
        headers[..., None],  # pad sample dim
        live_mask[..., None],  # pad sample dim
        file_root=file_root,
        segy_factory=segy_factory,
        drop_axis=-1,
    )

    # Recursively combine SEG-Y files from last (fastest) consecutive dimension
    # to first (slowest) dimension. End result will be the blocks with the
    # size of the outermost dimension in ascending order.
    consecutive_dim_index = find_trailing_ones_index(is_block_live.numblocks)
    while consecutive_dim_index != 1:
        current_chunks = is_block_live.chunksize

        prefix_dim = consecutive_dim_index - 1
        prefix_chunks = current_chunks[:prefix_dim]
        new_chunks = prefix_chunks + (-1,) * (len(current_chunks) - prefix_dim)

        is_block_live = map_blocks(
            segy_trace_concat,
            is_block_live.rechunk(new_chunks),
            consecutive_dim_index,
            file_root,
            drop_axis=-1,
        )

        consecutive_dim_index -= 1

    return is_block_live
