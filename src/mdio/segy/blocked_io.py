"""Functions for doing blocked I/O from SEG-Y."""


from __future__ import annotations

import multiprocessing as mp
from itertools import repeat

import numpy as np
from psutil import cpu_count
from segyio.tracefield import keys as segy_hdr_keys
from tqdm.auto import tqdm
from zarr import Blosc
from zarr import Group

from mdio.core import Grid
from mdio.core.indexing import ChunkIterator
from mdio.segy._workers import trace_worker_map


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
    parallel_inputs = zip(
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

    glob_count, glob_sum, glob_sum_square, glob_min, glob_max = zip(*chunk_stats)

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
