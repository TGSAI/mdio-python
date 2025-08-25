"""Low level workers for parsing and writing SEG-Y to Zarr."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import cast

import numpy as np
from segy import SegyFile

from mdio.schemas import ScalarType

if TYPE_CHECKING:
    from segy.arrays import HeaderArray
    from segy.config import SegySettings
    from segy.schema import SegySpec
    from xarray import Dataset as xr_Dataset
    from zarr import Array as zarr_Array

    from mdio.core.storage_location import StorageLocation


from mdio.constants import UINT32_MAX
from mdio.schemas.v1.dataset_serializer import _get_fill_value
from mdio.schemas.v1.stats import CenteredBinHistogram
from mdio.schemas.v1.stats import SummaryStatistics


class SegyFileArguments(TypedDict):
    """Arguments to open SegyFile instance creation."""

    url: str
    spec: SegySpec | None
    settings: SegySettings | None


def header_scan_worker(
    segy_kw: SegyFileArguments, trace_range: tuple[int, int], subset: list[str] | None = None
) -> HeaderArray:
    """Header scan worker.

    If SegyFile is not open, it can either accept a path string or a handle that was opened in
    a different context manager.

    Args:
        segy_kw: Arguments to open SegyFile instance.
        trace_range: Tuple consisting of the trace ranges to read.
        subset: List of header names to filter and keep.

    Returns:
        HeaderArray parsed from SEG-Y library.
    """
    segy_file = SegyFile(**segy_kw)

    slice_ = slice(*trace_range)

    cloud_native_mode = os.getenv("MDIO__IMPORT__CLOUD_NATIVE", default="False")

    if cloud_native_mode.lower() in {"true", "1"}:
        trace_header = segy_file.trace[slice_].header
    else:
        trace_header = segy_file.header[slice_]

    if subset is not None:
        trace_header = trace_header[subset]

    # Get non-void fields from dtype and copy to new array for memory efficiency
    fields = trace_header.dtype.fields
    non_void_fields = [(name, dtype) for name, (dtype, _) in fields.items()]
    new_dtype = np.dtype(non_void_fields)

    # Copy to non-padded memory, ndmin is to handle the case where there is 1 trace in block
    # (singleton) so we can concat and assign stuff later.
    trace_header = np.array(trace_header, dtype=new_dtype, ndmin=1)

    return cast("HeaderArray", trace_header)


def trace_worker(  # noqa: PLR0913
    segy_kw: SegyFileArguments,
    output_location: StorageLocation,
    data_variable_name: str,
    region: dict[str, slice],
    grid_map: zarr_Array,
    dataset: xr_Dataset,
) -> SummaryStatistics | None:
    """Writes a subset of traces from a region of the dataset of Zarr file.

    Args:
        segy_kw: Arguments to open SegyFile instance.
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
    if not dataset.trace_mask.any():
        return None

    # Open the SEG-Y file in every new process / spawned worker since the
    # open file handles cannot be shared across processes.
    segy_file = SegyFile(**segy_kw)

    not_null = grid_map != UINT32_MAX

    live_trace_indexes = grid_map[not_null].tolist()
    traces = segy_file.trace[live_trace_indexes]

    header_key = "headers"

    # Get subset of the dataset that has not yet been saved
    # The headers might not be present in the dataset
    worker_variables = [data_variable_name]
    if header_key in dataset.data_vars:  # Keeping the `if` here to allow for more worker configurations
        worker_variables.append(header_key)

    ds_to_write = dataset[worker_variables]

    if header_key in worker_variables:
        # Create temporary array for headers with the correct shape
        # TODO(BrianMichell): Implement this better so that we can enable fill values without changing the code. #noqa: TD003
        tmp_headers = np.zeros_like(dataset[header_key])
        tmp_headers[not_null] = traces.header
        ds_to_write[header_key][:] = tmp_headers

    data_variable = ds_to_write[data_variable_name]
    fill_value = _get_fill_value(ScalarType(data_variable.dtype.name))
    tmp_samples = np.full_like(data_variable, fill_value=fill_value)
    tmp_samples[not_null] = traces.sample
    ds_to_write[data_variable_name][:] = tmp_samples

    ds_to_write.to_zarr(output_location.uri, region=region, mode="r+", write_empty_chunks=False, zarr_format=2)

    histogram = CenteredBinHistogram(bin_centers=[], counts=[])
    return SummaryStatistics(
        count=traces.sample.size,
        min=traces.sample.min(),
        max=traces.sample.max(),
        sum=traces.sample.sum(),
        sum_squares=(traces.sample**2).sum(),
        histogram=histogram,
    )
