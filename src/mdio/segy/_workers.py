"""Low level workers for parsing and writing SEG-Y to Zarr."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import cast

import numpy as np
from segy import SegyFile

from mdio.api.io import to_mdio
from mdio.builder.schemas.dtype import ScalarType
from mdio.segy._disaster_recovery_wrapper import SegyFileTraceDataWrapper

if TYPE_CHECKING:
    from segy.arrays import HeaderArray
    from segy.config import SegySettings
    from segy.schema import SegySpec
    from upath import UPath
    from xarray import Dataset as xr_Dataset
    from zarr import Array as zarr_Array

from xarray import Variable
from zarr.core.config import config as zarr_config

from mdio.builder.schemas.v1.stats import CenteredBinHistogram
from mdio.builder.schemas.v1.stats import SummaryStatistics
from mdio.builder.xarray_builder import _get_fill_value
from mdio.constants import fill_value_map


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
        # struct field selection needs a list, not a tuple; a subset is a tuple from the template.
        trace_header = trace_header[list(subset)]

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
    output_path: UPath,
    data_variable_name: str,
    region: dict[str, slice],
    grid_map: zarr_Array,
    dataset: xr_Dataset,
) -> SummaryStatistics | None:
    """Writes a subset of traces from a region of the dataset of Zarr file.

    Args:
        segy_kw: Arguments to open SegyFile instance.
        output_path: Universal Path for the output Zarr dataset
            (e.g. local file path or cloud storage URI) the location
            also includes storage options for cloud storage.
        data_variable_name: Name of the data variable to write.
        region: Region of the dataset to write to.
        grid_map: Zarr array mapping live traces to their positions in the dataset.
        dataset: Xarray dataset containing the data to write.

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
    segy_file = SegyFile(**segy_kw)

    # Setting the zarr config to 1 thread to ensure we honor the `MDIO__IMPORT__MAX_WORKERS` environment variable.
    # The Zarr 3 engine utilizes multiple threads. This can lead to resource contention and unpredictable memory usage.
    zarr_config.set({"threading.max_workers": 1})

    live_trace_indexes = local_grid_map[not_null].tolist()

    header_key = "headers"
    raw_header_key = "raw_headers"

    # Get subset of the dataset that has not yet been saved
    # The headers might not be present in the dataset
    worker_variables = [data_variable_name]
    if header_key in dataset.data_vars:  # Keeping the `if` here to allow for more worker configurations
        worker_variables.append(header_key)
    if raw_header_key in dataset.data_vars:
        worker_variables.append(raw_header_key)

    # traces = segy_file.trace[live_trace_indexes]
    # Raw headers are not intended to remain as a feature of the SEGY ingestion.
    # For that reason, we have wrapped the accessors to provide an interface that can be removed
    # and not require additional changes to the below code.
    # NOTE: The `raw_header_key` code block should be removed in full as it will become dead code.
    traces = SegyFileTraceDataWrapper(segy_file, live_trace_indexes)

    ds_to_write = dataset[worker_variables]

    if raw_header_key in worker_variables:
        tmp_raw_headers = np.zeros_like(dataset[raw_header_key])
        tmp_raw_headers[not_null] = traces.raw_header

        ds_to_write[raw_header_key] = Variable(
            ds_to_write[raw_header_key].dims,
            tmp_raw_headers,
            attrs=ds_to_write[raw_header_key].attrs,
            encoding=ds_to_write[raw_header_key].encoding,  # Not strictly necessary, but safer than not doing it.
        )

    if header_key in worker_variables:
        # TODO(BrianMichell): Implement this better so that we can enable fill values without changing the code
        # https://github.com/TGSAI/mdio-python/issues/584
        tmp_headers = np.zeros_like(dataset[header_key])
        tmp_headers[not_null] = traces.header
        # Create a new Variable object to avoid copying the temporary array
        # The ideal solution is to use `ds_to_write[header_key][:] = tmp_headers`
        # but Xarray appears to be copying memory instead of doing direct assignment.
        # TODO(BrianMichell): #614 Look into this further.
        # https://github.com/TGSAI/mdio-python/issues/584
        ds_to_write[header_key] = Variable(
            ds_to_write[header_key].dims,
            tmp_headers,
            attrs=ds_to_write[header_key].attrs,
            encoding=ds_to_write[header_key].encoding,  # Not strictly necessary, but safer than not doing it.
        )

    data_variable = ds_to_write[data_variable_name]
    fill_value = _get_fill_value(ScalarType(data_variable.dtype.name))
    tmp_samples = np.full_like(data_variable, fill_value=fill_value)
    tmp_samples[not_null] = traces.sample

    # TODO(BrianMichell): #614 Look into this further.
    # https://github.com/TGSAI/mdio-python/issues/584
    ds_to_write[data_variable_name] = Variable(
        ds_to_write[data_variable_name].dims,
        tmp_samples,
        attrs=ds_to_write[data_variable_name].attrs,
        encoding=ds_to_write[data_variable_name].encoding,  # Not strictly necessary, but safer than not doing it.
    )

    to_mdio(ds_to_write, output_path=output_path, region=region, mode="r+")

    nonzero_samples = np.ma.masked_values(traces.sample, 0, copy=False)
    histogram = CenteredBinHistogram(bin_centers=[], counts=[])
    return SummaryStatistics(
        count=nonzero_samples.count(),
        min=nonzero_samples.min(),
        max=nonzero_samples.max(),
        sum=nonzero_samples.sum(dtype="float64"),
        sum_squares=(np.ma.power(nonzero_samples, 2).sum(dtype="float64")),
        histogram=histogram,
    )
