"""Low level workers for parsing and writing SEG-Y to Zarr."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import cast

import numpy as np

from mdio.core.storage_location import StorageLocation


if TYPE_CHECKING:
    from segy import SegyFile
    from segy.arrays import HeaderArray
    from xarray import Dataset as xr_Dataset
    from zarr import Array as zarr_Array

from mdio.constants import UINT32_MAX
from mdio.schemas.v1.stats import CenteredBinHistogram
from mdio.schemas.v1.stats import SummaryStatistics


def header_scan_worker(segy_file: SegyFile, trace_range: tuple[int, int]) -> HeaderArray:
    """Header scan worker.

    If SegyFile is not open, it can either accept a path string or a handle that was opened in
    a different context manager.

    Args:
        segy_file: SegyFile instance.
        trace_range: Tuple consisting of the trace ranges to read.

    Returns:
        HeaderArray parsed from SEG-Y library.
    """
    slice_ = slice(*trace_range)

    cloud_native_mode = os.getenv("MDIO__IMPORT__CLOUD_NATIVE", default="False")

    if cloud_native_mode.lower() in {"true", "1"}:
        trace_header = segy_file.trace[slice_].header
    else:
        trace_header = segy_file.header[slice_]

    # Get non-void fields from dtype and copy to new array for memory efficiency
    fields = trace_header.dtype.fields
    non_void_fields = [(name, dtype) for name, (dtype, _) in fields.items()]
    new_dtype = np.dtype(non_void_fields)

    # Copy to non-padded memory, ndmin is to handle the case where there is 1 trace in block
    # (singleton) so we can concat and assign stuff later.
    trace_header = np.array(trace_header, dtype=new_dtype, ndmin=1)

    return cast("HeaderArray", trace_header)


def trace_worker_v1(  # noqa: PLR0913
    segy_file: SegyFile,
    output_location: StorageLocation,
    data_variable_name: str,
    region: dict[str, slice],
    grid_map: zarr_Array,
    dataset: xr_Dataset,
) -> SummaryStatistics | None:
    """Writes a subset of traces from a region of the dataset of Zarr file.

    Args:
        segy_file: SegyFile instance.
        output_location: StorageLocation for the output Zarr dataset
            (e.g. local file path or cloud storage URI) the location
            also includes storage options for cloud storage.
        data_variable_name: Name of the data variable to write.
        region: Region of the dataset to write to.
        grid_map: Zarr array mapping live traces to their positions in the dataset.
        dataset: Xarray dataset containing the data to write.

    Returns:
        SummaryStatistics object containing statistics about the written traces.
    """
    if dataset.trace_mask.sum() == 0:
        return None

    not_null = grid_map != UINT32_MAX

    live_trace_indexes = grid_map[not_null].tolist()
    traces = segy_file.trace[live_trace_indexes]

    # Get subset of the dataset that has not yet been saved
    # The headers might not be present in the dataset
    # TODO(Dmitriy Repin): Check, should we overwrite the 'dataset' instead to save the memory
    # https://github.com/TGSAI/mdio-python/issues/584
    if "headers" in dataset.data_vars:
        ds_to_write = dataset[[data_variable_name, "headers"]]
        ds_to_write = ds_to_write.reset_coords()

        ds_to_write["headers"].data[not_null] = traces.header
        ds_to_write["headers"].data[~not_null] = 0
    else:
        ds_to_write = dataset[[data_variable_name]]
        ds_to_write = ds_to_write.reset_coords()

    ds_to_write[data_variable_name].data[not_null] = traces.sample

    out_path = output_location.uri
    ds_to_write.to_zarr(out_path, region=region, mode="r+", write_empty_chunks=False, zarr_format=2)

    histogram = CenteredBinHistogram(bin_centers=[], counts=[])
    return SummaryStatistics(
        count=traces.sample.size,
        min=traces.sample.min(),
        max=traces.sample.max(),
        sum=traces.sample.sum(),
        # TODO(Altay): Look at how to do the sum squares statistic correctly
        # https://github.com/TGSAI/mdio-python/issues/581
        sum_squares=(traces.sample**2).sum(),
        histogram=histogram,
    )
