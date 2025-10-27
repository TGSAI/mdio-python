"""Low level workers for parsing and writing SEG-Y to Zarr."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from segy.arrays import HeaderArray

from mdio.api.io import _normalize_storage_options
from mdio.segy._raw_trace_wrapper import SegyFileRawTraceWrapper
from mdio.segy.file import SegyFileArguments
from mdio.segy.file import SegyFileWrapper

if TYPE_CHECKING:
    from upath import UPath
    from zarr import Array as zarr_Array

from zarr import open_group as zarr_open_group
from zarr.core.config import config as zarr_config

from mdio.builder.schemas.v1.stats import CenteredBinHistogram
from mdio.builder.schemas.v1.stats import SummaryStatistics
from mdio.constants import fill_value_map

logger = logging.getLogger(__name__)


def header_scan_worker(
    segy_file_kwargs: SegyFileArguments,
    trace_range: tuple[int, int],
    subset: tuple[str, ...] | None = None,
) -> HeaderArray:
    """Header scan worker.

    If SegyFile is not open, it can either accept a path string or a handle that was opened in
    a different context manager.

    Args:
        segy_file_kwargs: Arguments to open SegyFile instance.
        trace_range: Tuple consisting of the trace ranges to read.
        subset: Tuple of header names to filter and keep.

    Returns:
        HeaderArray parsed from SEG-Y library.
    """
    segy_file = SegyFileWrapper(**segy_file_kwargs)

    slice_ = slice(*trace_range)

    cloud_native_mode = os.getenv("MDIO__IMPORT__CLOUD_NATIVE", default="False")

    if cloud_native_mode.lower() in {"true", "1"}:
        trace_header = segy_file.trace[slice_].header
    else:
        trace_header = segy_file.header[slice_]

    if subset is not None:
        # struct field selection needs a list, not a tuple; a subset is a tuple from the template.
        trace_header = trace_header[list(subset)]

    # Get non-void fields from dtype and copy to new array for memory efficiency
    fields = trace_header.dtype.fields
    non_void_fields = [(name, dtype) for name, (dtype, _) in fields.items()]
    new_dtype = np.dtype(non_void_fields)

    # Copy to non-padded memory, ndmin is to handle the case where there is 1 trace in block
    # (singleton) so we can concat and assign stuff later.
    trace_header = np.array(trace_header, dtype=new_dtype, ndmin=1)

    return HeaderArray(trace_header)  # wrap back so we can use aliases


def trace_worker(  # noqa: PLR0913
    segy_file_kwargs: SegyFileArguments,
    output_path: UPath,
    data_variable_name: str,
    region: dict[str, slice],
    grid_map: zarr_Array,
) -> SummaryStatistics | None:
    """Writes a subset of traces from a region of the dataset of Zarr file.

    Args:
        segy_file_kwargs: Arguments to open SegyFile instance.
        output_path: Universal Path for the output Zarr dataset
            (e.g. local file path or cloud storage URI) the location
            also includes storage options for cloud storage.
        data_variable_name: Name of the data variable to write.
        region: Region of the dataset to write to.
        grid_map: Zarr array mapping live traces to their positions in the dataset.

    Returns:
        SummaryStatistics object containing statistics about the written traces.
    """
    region_slices = tuple(region.values())
    local_grid_map = grid_map[region_slices[:-1]]  # minus last (vertical) axis

    # The dtype.max is the sentinel value for the grid map.
    # Normally, this is uint32, but some grids need to be promoted to uint64.
    not_null = local_grid_map != fill_value_map.get(local_grid_map.dtype.name)
    if not not_null.any():
        return None

    # Open the SEG-Y file in this process since the open file handles cannot be shared across processes.
    segy_file = SegyFileWrapper(**segy_file_kwargs)

    # Setting the zarr config to 1 thread to ensure we honor the `MDIO__IMPORT__MAX_WORKERS` environment variable.
    # The Zarr 3 engine utilizes multiple threads. This can lead to resource contention and unpredictable memory usage.
    zarr_config.set({"threading.max_workers": 1})

    live_trace_indexes = local_grid_map[not_null].tolist()

    # Open the zarr group to write directly
    storage_options = _normalize_storage_options(output_path)
    zarr_group = zarr_open_group(output_path.as_posix(), mode="r+", storage_options=storage_options)

    header_key = "headers"
    raw_header_key = "raw_headers"

    # Check which variables exist in the zarr store
    available_arrays = list(zarr_group.array_keys())

    # traces = segy_file.trace[live_trace_indexes]
    # Raw headers are not intended to remain as a feature of the SEGY ingestion.
    # For that reason, we have wrapped the accessors to provide an interface that can be removed
    # and not require additional changes to the below code.
    # NOTE: The `raw_header_key` code block should be removed in full as it will become dead code.
    traces = SegyFileRawTraceWrapper(segy_file, live_trace_indexes)

    # Compute slices once (headers exclude sample dimension)
    header_region_slices = region_slices[:-1]  # Exclude sample dimension

    full_shape = tuple(s.stop - s.start for s in region_slices)
    header_shape = tuple(s.stop - s.start for s in header_region_slices)

    # Write raw headers if they exist
    # Headers only have spatial dimensions (no sample dimension)
    if raw_header_key in available_arrays:
        raw_header_array = zarr_group[raw_header_key]
        tmp_raw_headers = np.full(header_shape, raw_header_array.fill_value)
        tmp_raw_headers[not_null] = traces.raw_header
        raw_header_array[header_region_slices] = tmp_raw_headers

    # Write headers if they exist
    # Headers only have spatial dimensions (no sample dimension)
    if header_key in available_arrays:
        header_array = zarr_group[header_key]
        tmp_headers = np.full(header_shape, header_array.fill_value)
        tmp_headers[not_null] = traces.header
        header_array[header_region_slices] = tmp_headers

    # Write the data variable
    data_array = zarr_group[data_variable_name]
    tmp_samples = np.full(full_shape, data_array.fill_value)
    tmp_samples[not_null] = traces.sample
    data_array[region_slices] = tmp_samples

    nonzero_samples = np.ma.masked_values(traces.sample, 0, copy=False)

    nonzero_count = nonzero_samples.count()
    if nonzero_count == 0:
        # Return None to avoid calculating a NaN in sum_squares
        return None

    histogram = CenteredBinHistogram(bin_centers=[], counts=[])
    return SummaryStatistics(
        count=nonzero_count,
        min=nonzero_samples.min(),
        max=nonzero_samples.max(),
        sum=nonzero_samples.sum(dtype="float64"),
        sum_squares=(np.ma.power(nonzero_samples, 2).sum(dtype="float64")),
        histogram=histogram,
    )
