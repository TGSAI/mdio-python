"""Functions for doing blocked I/O from SEG-Y."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr
from dask.array import Array
from dask.array import map_blocks
from segy import SegyFile
from tqdm.auto import tqdm
from zarr import open_group as zarr_open_group

from mdio.api.io import _normalize_storage_options
from mdio.builder.schemas.v1.stats import CenteredBinHistogram
from mdio.builder.schemas.v1.stats import SummaryStatistics
from mdio.constants import ZarrFormat
from mdio.core.config import MDIOSettings
from mdio.core.indexing import ChunkIterator
from mdio.segy._workers import trace_worker
from mdio.segy.creation import SegyPartRecord
from mdio.segy.creation import concat_files
from mdio.segy.creation import serialize_to_segy_stack
from mdio.segy.utilities import find_trailing_ones_index

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from segy import SegyFactory
    from upath import UPath
    from xarray import Dataset as xr_Dataset
    from zarr import Array as zarr_Array

    from mdio.segy.file import SegyFileArguments


def _create_stats() -> SummaryStatistics:
    histogram = CenteredBinHistogram(bin_centers=[], counts=[])
    return SummaryStatistics(count=0, min=0, max=0, sum=0, sum_squares=0, histogram=histogram)


def _update_stats(final_stats: SummaryStatistics, partial_stats: SummaryStatistics) -> None:
    final_stats.count += partial_stats.count
    final_stats.min = min(final_stats.min, partial_stats.min)
    final_stats.max = max(final_stats.max, partial_stats.max)
    final_stats.sum += partial_stats.sum
    final_stats.sum_squares += partial_stats.sum_squares


def to_zarr(  # noqa: PLR0913, PLR0915
    segy_file_kwargs: SegyFileArguments,
    output_path: UPath,
    grid_map: zarr_Array,
    dataset: xr_Dataset,
    data_variable_name: str,
) -> SummaryStatistics:
    """Blocked I/O from SEG-Y to chunked `xarray.Dataset`.

    Args:
        segy_file_kwargs: SEG-Y file arguments.
        output_path: Output universal path for the output MDIO dataset.
        grid_map: Zarr array with grid map for the traces.
        dataset: Handle for xarray.Dataset we are writing trace data
        data_variable_name: Name of the data variable in the dataset.

    Returns:
        None
    """
    settings = MDIOSettings()

    data = dataset[data_variable_name]

    final_stats = _create_stats()

    data_variable_chunks = data.encoding.get("chunks")
    worker_chunks = data_variable_chunks[:-1] + (data.shape[-1],)  # un-chunk sample axis
    chunk_iter = ChunkIterator(shape=data.shape, chunks=worker_chunks, dim_names=data.dims)
    num_chunks = chunk_iter.num_chunks

    zarr_format = zarr.config.get("default_zarr_format")

    # Open zarr group once in main process
    storage_options = _normalize_storage_options(output_path)
    zarr_group = zarr_open_group(
        output_path.as_posix(),
        mode="r+",
        storage_options=storage_options,
        use_consolidated=zarr_format == ZarrFormat.V2,
    )

    # Get array handles from the opened group
    data_array = zarr_group[data_variable_name]
    header_array = zarr_group.get("headers")
    raw_header_array = zarr_group.get("raw_headers")

    # For Unix async writes with s3fs/fsspec & multiprocessing, use 'spawn' instead of default
    # 'fork' to avoid deadlocks on cloud stores. Slower but necessary. Default on Windows.
    num_workers = min(num_chunks, settings.import_cpus)
    context = mp.get_context("spawn")

    # Use initializer to open segy file once per worker
    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=context,
    )

    segy_file = SegyFile(**segy_file_kwargs)

    with executor:
        futures = []
        for region in chunk_iter:
            # Pass zarr array handles directly to workers
            future = executor.submit(
                trace_worker,
                segy_file,
                data_array,
                header_array,
                raw_header_array,
                region,
                grid_map,
            )
            futures.append(future)

        iterable = tqdm(
            as_completed(futures),
            total=num_chunks,
            unit="block",
            desc="Ingesting traces",
        )

        for future in iterable:
            result = future.result()
            if result is not None:
                _update_stats(final_stats, result)

    # Update statistics using the already-open zarr group
    # Xarray doesn't directly support incremental attribute updates when appending to an existing Zarr store.
    # HACK: We will update the array attribute using zarr's API directly.
    attr_json = final_stats.model_dump_json()
    zarr_group[data_variable_name].attrs.update({"statsV1": attr_json})

    if zarr.config.get("default_zarr_format") == ZarrFormat.V2:
        zarr.consolidate_metadata(zarr_group.store)

    return final_stats


def segy_record_concat(
    block_records: NDArray,
    file_root: str,
    block_info: dict | None = None,
) -> NDArray:
    """Concatenate partial ordered SEG-Y blocks on disk.

    It will take an ND array SegyPartRecords. Goal is to preserve the global order of traces when
    merging files. Order is assumed to be correct at the block level (last dimension).

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
        msg = "block_info is required for global index computation."
        raise ValueError(msg)

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

        global_index = tuple(block_start[i] + rec_index[i] for i in range(len(record_shape)))
        record_id_str = "/".join(map(str, global_index))
        record_file_path = Path(file_root) / f"{record_id_str}.bin"

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

    Blocks are written out in parallel via multiple workers, and then adjacent blocks are tracked
    and merged on disk via the `segy_trace_concat` function. The adjacent are hierarchically
    merged, and it preserves order.

    Assume array with shape (4, 3, 2) with chunk sizes (1, 1, 2). The chunk indices for this
    array would be:

    (0, 0, 0) (0, 1, 0) (0, 2, 0)
    (1, 0, 0) (1, 1, 0) (1, 2, 0)
    (2, 0, 0) (2, 1, 0) (2, 2, 0)
    (3, 0, 0) (3, 1, 0) (3, 2, 0)

    Let's rename them to this for convenience:

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

    During all the processing here, we keep track of logical indices of chunks and written files
    so we can correctly combine them. The above algorithm generalizes to higher dimensions.

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
    reduce_axes = tuple(i for i in range(non_consecutive_axes - 1, len(num_blocks)) if num_blocks[i] == 1)

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

    # Recursively combine SEG-Y files from fastest consecutive dimension to first dimension.
    # End result will be the blocks with the size of the outermost dimension in ascending order.
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
