"""Low level workers for parsing and writing SEG-Y to Zarr."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import TypedDict

import numpy as np
from segy import SegyFile
from segy.arrays import HeaderArray

from mdio.api.io import _normalize_storage_options
from mdio.segy._raw_trace_wrapper import SegyFileRawTraceWrapper
from mdio.segy.scalar import _get_coordinate_scalar

if TYPE_CHECKING:
    from segy.config import SegyFileSettings
    from segy.config import SegyHeaderOverrides
    from segy.schema import SegySpec
    from upath import UPath
    from zarr import Array as zarr_Array

from zarr import open_group as zarr_open_group
from zarr.core.config import config as zarr_config

from mdio.builder.schemas.v1.stats import CenteredBinHistogram
from mdio.builder.schemas.v1.stats import SummaryStatistics
from mdio.constants import fill_value_map

if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class SegyFileArguments(TypedDict):
    """Arguments to open SegyFile instance creation."""

    url: str
    spec: SegySpec | None
    settings: SegyFileSettings | None
    header_overrides: SegyHeaderOverrides | None


def header_scan_worker(
    segy_file_kwargs: SegyFileArguments,
    trace_range: tuple[int, int],
    subset: list[str] | None = None,
) -> HeaderArray:
    """Header scan worker.

    If SegyFile is not open, it can either accept a path string or a handle that was opened in
    a different context manager.

    Args:
        segy_file_kwargs: Arguments to open SegyFile instance.
        trace_range: Tuple consisting of the trace ranges to read.
        subset: List of header names to filter and keep.

    Returns:
        HeaderArray parsed from SEG-Y library.
    """
    segy_file = SegyFile(**segy_file_kwargs)

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
    segy_file = SegyFile(**segy_file_kwargs)

    # Setting the zarr config to 1 thread to ensure we honor the `MDIO__IMPORT__MAX_WORKERS` environment variable.
    # The Zarr 3 engine utilizes multiple threads. This can lead to resource contention and unpredictable memory usage.
    zarr_config.set({"threading.max_workers": 1})

    live_trace_indexes = local_grid_map[not_null].tolist()

    # Open the zarr group to write directly
    storage_options = _normalize_storage_options(output_path)
    zarr_group = zarr_open_group(
        output_path.as_posix(), mode="r+", storage_options=storage_options
    )

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

    # Write raw headers if they exist
    # Headers only have spatial dimensions (no sample dimension)
    if raw_header_key in available_arrays:
        zarr_array = zarr_group[raw_header_key]
        # Read existing data, modify live traces, write back
        # This avoids allocating a new array and is memory efficient
        tmp_raw_headers = zarr_array[header_region_slices]
        tmp_raw_headers[not_null] = traces.raw_header
        zarr_array[header_region_slices] = tmp_raw_headers

    # Write headers if they exist
    # Headers only have spatial dimensions (no sample dimension)
    if header_key in available_arrays:
        zarr_array = zarr_group[header_key]
        # Read existing data, modify live traces, write back
        tmp_headers = zarr_array[header_region_slices]
        tmp_headers[not_null] = traces.header
        zarr_array[header_region_slices] = tmp_headers

    # Write the data variable
    zarr_array = zarr_group[data_variable_name]
    # Read existing data, modify live traces, write back
    tmp_samples = zarr_array[region_slices]
    tmp_samples[not_null] = traces.sample
    zarr_array[region_slices] = tmp_samples

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


@dataclass
class SegyFileInfo:
    """SEG-Y file header information."""

    num_traces: int
    sample_labels: NDArray[np.int32]
    text_header: str
    binary_header_dict: dict
    raw_binary_headers: bytes
    coordinate_scalar: int


def info_worker(segy_file_kwargs: SegyFileArguments) -> SegyFileInfo:
    """Reads information from a SEG-Y file.

    Args:
        segy_file_kwargs: Arguments to open SegyFile instance.

    Returns:
        SegyFileInfo containing number of traces, sample labels, and header info.
    """
    segy_file = SegyFile(**segy_file_kwargs)
    num_traces = segy_file.num_traces
    sample_labels = segy_file.sample_labels

    text_header = segy_file.text_header

    # Get header information directly
    raw_binary_headers = segy_file.fs.read_block(
        fn=segy_file.url,
        offset=segy_file.spec.binary_header.offset,
        length=segy_file.spec.binary_header.itemsize,
    )

    # We read here twice, but it's ok for now. Only 400-bytes.
    binary_header_dict = segy_file.binary_header.to_dict()

    coordinate_scalar = _get_coordinate_scalar(segy_file)

    return SegyFileInfo(
        num_traces=num_traces,
        sample_labels=sample_labels,
        text_header=text_header,
        binary_header_dict=binary_header_dict,
        raw_binary_headers=raw_binary_headers,
        coordinate_scalar=coordinate_scalar,
    )
