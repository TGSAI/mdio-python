"""Functions for doing blocked I/O from SEG-Y."""


from __future__ import annotations

import multiprocessing as mp
from itertools import repeat

import numpy as np
from dask.array import Array
from dask.array import blockwise
from dask.array.reductions import _tree_reduce
from numpy.typing import NDArray
from psutil import cpu_count
from segyio.tracefield import keys as segy_hdr_keys
from tqdm.auto import tqdm
from zarr import Blosc
from zarr import Group

from mdio.core import Grid
from mdio.core.indexing import ChunkIterator
from mdio.segy._workers import trace_worker_map
from mdio.segy.byte_utils import ByteOrder
from mdio.segy.byte_utils import Dtype
from mdio.segy.creation import concat_files
from mdio.segy.creation import write_to_segy_stack


try:
    import zfpy  # Base library
    from zarr import ZFPY  # Codec

except ImportError:
    ZFPY = None
    zfpy = None

# Globals
NUM_CORES = cpu_count(logical=False)


def to_zarr(
    segy_path: str,
    segy_endian: str,
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
        segy_path: Path to the input SEG-Y file
        segy_endian: Endianness of the input SEG-Y.
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
        **kwargs,
    )

    # Here we read the byte locations of the rev.1 SEG-Y standard as defined in segyio
    # We skip the last two, because segyio doesn't read it when we ask for traces
    rev1_bytes = list(segy_hdr_keys.values())[:-2]

    # Then we diff the byte locations to get lengths of headers. Last one has to be
    # manually added because it is forward diff.
    header_byte_loc = list(map(str, rev1_bytes))
    header_byte_len = [
        rev1_bytes[idx + 1] - rev1_bytes[idx] for idx in range(len(header_byte_loc) - 1)
    ]
    header_byte_len += [2]  # Length of last TraceField (SourceMeasurementUnit)

    # Make numpy.dtype
    # We assume either 16-bit or 32-bit signed integers (per SEG-Y standard).
    # In numpy these are 'i2' (aka 'int16') or 'i4' (aka 'int32')
    header_dtype = {
        "names": header_byte_loc,
        "formats": [f"i{length}" for length in header_byte_len],
    }
    header_dtype = np.dtype(header_dtype)

    header_array = metadata_root.create_dataset(
        name="_".join([name, "trace_headers"]),
        shape=grid.shape[:-1],  # Same spatial shape as data
        chunks=chunks[:-1],  # Same spatial chunks as data
        compressor=header_compressor,
        dtype=header_dtype,
        dimension_separator="/",
    )

    # Initialize chunk iterator (returns next chunk slice indices each iteration)
    chunker = ChunkIterator(trace_array, chunk_samples=False)
    num_chunks = len(chunker)

    # Setting all multiprocessing parameters.
    parallel_inputs = zip(  # noqa: B905
        repeat(segy_path),
        repeat(trace_array),
        repeat(header_array),
        repeat(grid),
        chunker,
        repeat(segy_endian),
    )

    # This is for Unix async writes to s3fs/fsspec, when using
    # multiprocessing. By default, Linux uses the 'fork' method.
    # 'spawn' is a little slower to spool up processes, but 'fork'
    # doesn't work. If you don't use this, processes get deadlocked
    # on cloud stores. 'spawn' is default in Windows.
    context = mp.get_context("spawn")

    # This is the chunksize for multiprocessing. Not to be confused
    # with Zarr chunksize.
    num_workers = min(num_chunks, NUM_CORES)
    pool_chunksize, extra = divmod(num_chunks, num_workers * 4)
    pool_chunksize += 1 if extra else pool_chunksize

    tqdm_kw = dict(unit="block", dynamic_ncols=True)
    with context.Pool(num_workers) as pool:
        # pool.imap is lazy
        lazy_work = pool.imap(
            func=trace_worker_map,
            iterable=parallel_inputs,
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


def segy_concat(
    partial_files: NDArray,
    axis: tuple[int] = None,
    keepdims: bool = None,
) -> NDArray:
    """Aggregate partial SEG-Y blocks on disk, preserving order.

    Used in conjunction with tree reduction. It will take an array
    of file names, which preserved the adjacency of blocks, and then
    combines adjacent blocks while flattening for SEG-Y.

    For `axis` and `keepdims` parameters, please see `dask.array.reduce`
    documentation.

    Args:
        partial_files: Array containing paths to parts of a SEG-Y row.
        axis: Which axes to concatenate on.
        keepdims: Keep the original dimensionality after merging.

    Returns:
        Concatenated file name array. Dimensions depend on `keepdims`.
    """
    concat_shape = partial_files.shape[0]
    concat_paths = np.full_like(partial_files, fill_value="missing", shape=concat_shape)

    # Fast path if all data in block is missing.
    if np.all(partial_files == "missing"):
        return np.expand_dims(concat_paths, axis) if keepdims else concat_paths

    # Flatten and concat section files to a single root file at first axis.
    for index, section_paths in enumerate(partial_files):
        section_paths = section_paths.ravel()
        section_missing = section_paths == "missing"

        if np.all(section_missing):
            continue

        section_valid_paths = np.extract(~section_missing, section_paths).tolist()
        section_concat_files = concat_files(section_valid_paths)
        concat_paths[index] = section_concat_files

    return np.expand_dims(concat_paths, axis) if keepdims else concat_paths


def to_segy(
    samples: Array,
    headers: Array,
    live_mask: Array,
    out_dtype: Dtype,
    out_byteorder: ByteOrder,
    file_root: str,
    axis: tuple[int] | None = None,
) -> Array:
    r"""Convert MDIO blocks to SEG-Y parts.

    This uses as a tree reduction algorithm. Blocks are written out
    in parallel via multiple workers, and then adjacent blocks are
    tracked and merged on disk via the `segy_concat` function. The
    adjacent are hierarchically merged, and it preserves order.

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

    The module will return file names associated with these
    concatenated files. Then they can be combined to form the
    sequence "abcdefghi" which is what we want.

    The above algorithm extrapolates to higher dimensions.

    Args:
        samples: Sample array.
        headers: Header array.
        live_mask: Live mask array.
        out_dtype: Desired type of output samples.
        out_dtype: Desired output data type.
        out_byteorder: Desired output data byte order.
        file_root: Root directory to write partial SEG-Y files.
        axis: Which axes to merge on. Excluding sample axis.

    Returns:
        Array containing final, flattened SEG-Y blocks.
    """
    # Map chunk across all blocks
    samp_inds = tuple(range(samples.ndim))
    meta_inds = tuple(range(headers.ndim))

    args = (samples, samp_inds)
    args += (headers, meta_inds)
    args += (live_mask, meta_inds)

    # Merge samples axis, append headers, and write block as stack of SEG-Ys.
    # Note: output is N-1 dimensional (meta_inds) because we merged samples.
    trace_files = blockwise(
        write_to_segy_stack,
        meta_inds,
        *args,
        file_root=file_root,
        out_dtype=out_dtype,
        out_byteorder=out_byteorder,
        concatenate=True,
    )

    result = _tree_reduce(
        trace_files,
        segy_concat,
        axis,
        keepdims=False,
        dtype=trace_files.dtype,
    )

    return result
