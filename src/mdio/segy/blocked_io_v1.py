
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from typing import Any

import numpy as np
from tqdm.auto import tqdm
import zarr
from psutil import cpu_count
from segy import SegyFile
from segy.arrays import HeaderArray

import xarray as xr
from xarray import Dataset as xr_Dataset
from xarray import DataArray as xr_DataArray

from mdio.constants import UINT32_MAX
from mdio.core.indexing_v1 import ChunkIterator, ShapeAndChunks
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.stats import CenteredBinHistogram, SummaryStatistics
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.segy.utilities import segy_export_rechunker
from tests.integration.test_segy_import_export_masked import Dimension

def _create_stats() -> SummaryStatistics:
    histogram = CenteredBinHistogram(bin_centers=[], counts=[])
    stats = SummaryStatistics(
        count=0, min=0, max=0, sum=0, sum_squares=0, histogram=histogram
    )
    return stats


def _update_stats(final_stats: SummaryStatistics, partial_stats: SummaryStatistics) -> None:
    final_stats.count += partial_stats.count
    final_stats.min = min(final_stats.min, partial_stats.min)
    final_stats.max = min(final_stats.max, partial_stats.max)
    final_stats.sum += partial_stats.sum
    final_stats.sum_squares += partial_stats.sum_squares


def _create_executor(num_chunks: int)-> ProcessPoolExecutor:
    default_cpus = cpu_count(logical=True)
    num_cpus = int(os.getenv("MDIO__IMPORT__CPU_COUNT", default_cpus))
    num_workers = min(num_chunks, num_cpus)
    # For Unix async writes with s3fs/fsspec & multiprocessing, use 'spawn' instead of default
    # 'fork' to avoid deadlocks on cloud stores. Slower but necessary. Default on Windows.
    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=context)
    return executor


def _make_shape_and_chunks(data: xr_DataArray) -> ShapeAndChunks:
    """Get shape and chunks from the template."""
    # The function returns a tuple of tuples
    optimal_chunks = segy_export_rechunker(
        shape=data.shape,
        chunks= data.encoding.get("chunks"),
        dtype=data.dtype)
    # Unroll it to a tuple
    optimal_chunks = sum(optimal_chunks, ())
    return ShapeAndChunks(shape=data.shape, chunks=optimal_chunks)


def _traces_to_zarr(  # noqa: PLR0913
    segy_file: SegyFile,
    out_path: str,
    data_variable_name: str,
    region: dict[str, slice],
    grid_map: zarr.Array,
    dataset: xr_Dataset
) -> SummaryStatistics | None:
    """Read a subset of traces and write to region of Zarr file."""
    if dataset.trace_mask.sum() == 0:
        return None
    
    not_null = grid_map != UINT32_MAX

    live_trace_indexes = grid_map[not_null].tolist()
    traces = segy_file.trace[live_trace_indexes]

    # Get subset of the dataset that has not yet been saved
    # The headers might not be present in the dataset
    if "headers" in dataset.data_vars:
        ds_to_write = dataset[[data_variable_name, "headers"]]
        ds_to_write = dataset.reset_coords()
        ds_to_write = dataset.drop_vars(["trace_mask"])
        ds_to_write["headers"].data[not_null] = traces.header
        ds_to_write["headers"].data[~not_null] = 0
        # BUG: Fails here with "IndexError: Boolean array with size 400 is not long enough for axis 0 with size 20"At the moment
        ds_to_write[data_variable_name].data[not_null] = traces.sample
        ds_to_write.to_zarr(out_path, region=region, mode="r+", write_empty_chunks=False)
    else:
        ds_to_write = dataset[[data_variable_name]]
        ds_to_write = dataset.reset_coords()
        ds_to_write = dataset.drop_vars(["trace_mask"])
        # Note: the dimension and coordinates variables have already been dropped
        # BUG: Fails here with "IndexError: Boolean array with size 400 is not long enough for axis 0 with size 20"
        #  ds_to_write[data_variable_name].data[not_null] = traces.sample
        # Fix it as following:
        data_var = ds_to_write[data_variable_name]
        data_var.data = traces.sample.reshape(data_var.shape)
        ds_to_write.to_zarr(out_path, region=region, mode="r+", write_empty_chunks=False)

    histogram = CenteredBinHistogram(bin_centers=[], counts=[])
    return SummaryStatistics(
        count=traces.sample.size,
        min=traces.sample.min(),
        max=traces.sample.max(),
        sum=traces.sample.sum(),
        sum_squares=(traces.sample**2).sum(),
        histogram=histogram,
    )


def to_zarr(  # noqa: PLR0913, PLR0915
    segy_file: SegyFile,
    out_path: str,
    grid_map: zarr.Array,
    dataset: xr_Dataset,
    data_variable_name: str
) -> SummaryStatistics:
    """Write data array."""

    data = dataset[data_variable_name]   
    # The last dimension is the vertical dimension
    index_keys = data.dims[:-1]

    final_stats = _create_stats()

    shape_and_chunks = _make_shape_and_chunks(data)
    chunk_iter = ChunkIterator(shape_and_chunks, data.dims, False)
    num_chunks = len(chunk_iter)
    executor = _create_executor(num_chunks=num_chunks)
    with executor:
        futures = []
        common_args = (segy_file, out_path, data_variable_name)
        for region in chunk_iter:
            index_slices = tuple(region[key] for key in index_keys)
            subset_args = (
                region,
                grid_map[index_slices],
                dataset.isel(region), 
            )
            future = executor.submit(_traces_to_zarr, *common_args, *subset_args)
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

    return final_stats