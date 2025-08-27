"""Conversion from SEG-Y to MDIO v1 format."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from segy import SegyFile
from segy.config import SegySettings
from segy.standards.codes import MeasurementSystem as segy_MeasurementSystem
from segy.standards.fields.trace import Rev0 as TraceHeaderFieldsRev0

from mdio.api.io import _normalize_path
from mdio.api.io import to_mdio
from mdio.constants import UINT32_MAX
from mdio.converters.exceptions import EnvironmentFormatError
from mdio.converters.exceptions import GridTraceCountError
from mdio.converters.exceptions import GridTraceSparsityError
from mdio.converters.type_converter import to_structured_type
from mdio.core.grid import Grid
from mdio.schemas.v1.dataset_serializer import to_xarray_dataset
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.segy import blocked_io
from mdio.segy.utilities import get_grid_plan

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from segy.arrays import HeaderArray as SegyHeaderArray
    from segy.schema import SegySpec
    from upath import UPath
    from xarray import Dataset as xr_Dataset

    from mdio.core.dimension import Dimension
    from mdio.schemas.v1.dataset import Dataset
    from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate

logger = logging.getLogger(__name__)


def grid_density_qc(grid: Grid, num_traces: int) -> None:
    """Quality control for sensible grid density during SEG-Y to MDIO conversion.

    This function checks the density of the proposed grid by comparing the total possible traces
    (`grid_traces`) to the actual number of traces in the SEG-Y file (`num_traces`). A warning is
    logged if the sparsity ratio (`grid_traces / num_traces`) exceeds a configurable threshold,
    indicating potential inefficiency or misconfiguration.

    The warning threshold is set via the environment variable `MDIO__GRID__SPARSITY_RATIO_WARN`
    (default 2), and the error threshold via `MDIO__GRID__SPARSITY_RATIO_LIMIT` (default 10). To
    suppress the exception (but still log warnings), set `MDIO_IGNORE_CHECKS=1`.

    Args:
        grid: The Grid instance to check.
        num_traces: Expected number of traces from the SEG-Y file.

    Raises:
        GridTraceSparsityError: If the sparsity ratio exceeds `MDIO__GRID__SPARSITY_RATIO_LIMIT`
            and `MDIO_IGNORE_CHECKS` is not set to a truthy value (e.g., "1", "true").
        EnvironmentFormatError: If `MDIO__GRID__SPARSITY_RATIO_WARN` or
            `MDIO__GRID__SPARSITY_RATIO_LIMIT` cannot be converted to a float.
    """
    # Calculate total possible traces in the grid (excluding sample dimension)
    grid_traces = np.prod(grid.shape[:-1], dtype=np.uint64)

    # Handle division by zero if num_traces is 0
    sparsity_ratio = float("inf") if num_traces == 0 else grid_traces / num_traces

    # Fetch and validate environment variables
    warning_ratio_env = os.getenv("MDIO__GRID__SPARSITY_RATIO_WARN", "2")
    error_ratio_env = os.getenv("MDIO__GRID__SPARSITY_RATIO_LIMIT", "10")
    ignore_checks_env = os.getenv("MDIO_IGNORE_CHECKS", "false").lower()
    ignore_checks = ignore_checks_env in ("1", "true", "yes", "on")

    try:
        warning_ratio = float(warning_ratio_env)
    except ValueError as e:
        raise EnvironmentFormatError("MDIO__GRID__SPARSITY_RATIO_WARN", "float") from e  # noqa: EM101

    try:
        error_ratio = float(error_ratio_env)
    except ValueError as e:
        raise EnvironmentFormatError("MDIO__GRID__SPARSITY_RATIO_LIMIT", "float") from e  # noqa: EM101

    # Check sparsity
    should_warn = sparsity_ratio > warning_ratio
    should_error = sparsity_ratio > error_ratio and not ignore_checks

    # Early return if everything is OK
    # Prepare message for warning or error
    if not should_warn and not should_error:
        return

    # Build warning / error message
    dims = dict(zip(grid.dim_names, grid.shape, strict=True))
    msg = (
        f"Ingestion grid is sparse. Sparsity ratio: {sparsity_ratio:.2f}. "
        f"Ingestion grid: {dims}. "
        f"SEG-Y trace count: {num_traces}, grid trace count: {grid_traces}."
    )
    for dim_name in grid.dim_names:
        dim_min = grid.get_min(dim_name)
        dim_max = grid.get_max(dim_name)
        msg += f"\n{dim_name} min: {dim_min} max: {dim_max}"

    # Log warning if sparsity exceeds warning threshold
    if should_warn:
        logger.warning(msg)

    # Raise error if sparsity exceeds error threshold and checks are not ignored
    if should_error:
        raise GridTraceSparsityError(grid.shape, num_traces, msg)


def _scan_for_headers(
    segy_file: SegyFile,
    template: AbstractDatasetTemplate,
    grid_overrides: dict[str, Any] | None = None,
) -> tuple[list[Dimension], SegyHeaderArray]:
    """Extract trace dimensions and index headers from the SEG-Y file.

    This is an expensive operation.
    It scans the SEG-Y file in chunks by using ProcessPoolExecutor
    """
    full_chunk_size = template.full_chunk_size
    segy_dimensions, chunk_size, segy_headers = get_grid_plan(
        segy_file=segy_file,
        return_headers=True,
        template=template,
        chunksize=full_chunk_size,
        grid_overrides=grid_overrides,
    )
    if full_chunk_size != chunk_size:
        # TODO(Dmitriy): implement grid overrides
        # https://github.com/TGSAI/mdio-python/issues/585
        # The returned 'chunksize' is used only for grid_overrides. We will need to use it when full
        # support for grid overrides is implemented
        err = "Support for changing full_chunk_size in grid overrides is not yet implemented"
        raise NotImplementedError(err)
    return segy_dimensions, segy_headers


def _build_and_check_grid(segy_dimensions: list[Dimension], segy_file: SegyFile, segy_headers: SegyHeaderArray) -> Grid:
    """Build and check the grid from the SEG-Y headers and dimensions.

    Args:
        segy_dimensions: List of of all SEG-Y dimensions to build grid from.
        segy_file: Instance of SegyFile to check for trace count.
        segy_headers: Headers read in from SEG-Y file for building the trace map.

    Returns:
        A grid instance populated with the dimensions and trace index map.

    Raises:
        GridTraceCountError: If number of traces in SEG-Y file does not match the parsed grid
    """
    grid = Grid(dims=segy_dimensions)
    grid_density_qc(grid, segy_file.num_traces)
    grid.build_map(segy_headers)
    # Check grid validity by comparing trace numbers
    if np.sum(grid.live_mask) != segy_file.num_traces:
        for dim_name in grid.dim_names:
            dim_min, dim_max = grid.get_min(dim_name), grid.get_max(dim_name)
            logger.warning("%s min: %s max: %s", dim_name, dim_min, dim_max)
        logger.warning("Ingestion grid shape: %s.", grid.shape)
        raise GridTraceCountError(np.sum(grid.live_mask), segy_file.num_traces)
    return grid


def _get_coordinates(
    grid: Grid,
    segy_headers: SegyHeaderArray,
    mdio_template: AbstractDatasetTemplate,
) -> tuple[list[Dimension], dict[str, SegyHeaderArray]]:
    """Get the data dim and non-dim coordinates from the SEG-Y headers and MDIO template.

    Select a subset of the segy_dimensions that corresponds to the MDIO dimensions
    The dimensions are ordered as in the MDIO template.
    The last dimension is always the vertical domain dimension

    Args:
        grid: Inferred MDIO grid for SEG-Y file.
        segy_headers: Headers read in from SEG-Y file.
        mdio_template: The MDIO template to use for the conversion.

    Raises:
        ValueError: If a dimension or coordinate name from the MDIO template is not found in
                    the SEG-Y headers.

    Returns:
        A tuple containing:
            - A list of dimension coordinates (1-D arrays).
            - A dict of non-dimension coordinates (str: N-D arrays).
    """
    dimensions_coords = []
    for dim_name in mdio_template.dimension_names:
        if dim_name not in grid.dim_names:
            err = f"Dimension '{dim_name}' was not found in SEG-Y dimensions."
            raise ValueError(err)
        dimensions_coords.append(grid.select_dim(dim_name))

    non_dim_coords: dict[str, SegyHeaderArray] = {}
    for coord_name in mdio_template.coordinate_names:
        if coord_name not in segy_headers.dtype.names:
            err = f"Coordinate '{coord_name}' not found in SEG-Y dimensions."
            raise ValueError(err)
        non_dim_coords[coord_name] = segy_headers[coord_name]

    return dimensions_coords, non_dim_coords


def populate_dim_coordinates(
    dataset: xr_Dataset, grid: Grid, drop_vars_delayed: list[str]
) -> tuple[xr_Dataset, list[str]]:
    """Populate the xarray dataset with dimension coordinate variables."""
    for dim in grid.dims:
        dataset[dim.name].values[:] = dim.coords
        drop_vars_delayed.append(dim.name)
    return dataset, drop_vars_delayed


def _check_dimensions_values_identical(arr: np.ndarray, axes_to_check: tuple[int, ...]) -> np.ndarray:
    """
    Check if all values along specified dimensions are identical for each
    sub-array defined by the other dimensions.

    Args:
        arr (np.ndarray): an N-dimensional array. For example, an array of all 'cdp_x' segy
            header values for coordinates "inline", "crossline", "offset", "azimuth".
        axes_to_check (tuple): A tuple of integers representing the axes to check for
            identical values. For example, (2, 3) would check the "offset", "azimuth"
            dimensions.

    Returns:
        bool: True indicates the all values in the dimensions selected by axes_to_check
        are identical, and False otherwise.
    """

    # Create a slicing tuple to get the first element along the axes to check
    full_slice = [slice(None)] * arr.ndim
    for axis in axes_to_check:
        full_slice[axis] = 0

    # Broadcast the first element along the specified axes for comparison
    first_element_slice = arr[tuple(full_slice)]

    # Add new axes to the slice to enable broadcasting against the original array
    for axis in axes_to_check:
        first_element_slice = np.expand_dims(first_element_slice, axis)

    # Compare the array with the broadcasted slice and use np.all()
    # to collapse the dimensions being checked
    identical = np.all(np.isclose(arr, first_element_slice), axis=axes_to_check)
    return np.all(identical).item()


def _populate_non_dim_coordinates(
    dataset: xr_Dataset,
    grid: Grid,
    coordinate_headers: dict[str, SegyHeaderArray],
    drop_vars_delayed: list[str],
) -> tuple[xr_Dataset, list[str]]:
    """Populate the xarray dataset with coordinate variables."""
    coord_is_good = {}
    # Load the grid map values into memory.
    # Q: Should we be using the trace_mask boolean array instead of grid map?
    grid_map_values = grid.map[:]
    for c_name, coord_headers_values in coordinate_headers.items():

        # In the case of Coca and some other templates, the coordinate header values, 
        # coord_headers_values, have a full set of dimensions (e.g. a 4-tuple of "inline", 
        # "crossline", "offset", "azimuth"), while the non-dimensional coordinates, (e.g., 
        # dataset["cdp_x"]) are defined over only a subset of the dimensions (e.g. 2-tuple of 
        # "inline", "crossline"). 
        # Thus, every coordinate 2-tuple has multiple duplicate values of the "cdp_x" coordinates
        # stored in coord_headers_values. Those needs to be reduced to a unique value. 
        # We assume (and check) that all the duplicate values are (near) identical.
        headers_dims = grid.dim_names[:-1]  # e.g.: "inline", "crossline", "offset", "azimuth"
        coord_dims = dataset[c_name].dims   # e.g.: "inline", "crossline"
        # This will create a temporary array in memory with the same shape as the coordinate
        # defined in the dataset. Since the coordinate variable has not yet been populated,
        # the temporary array will be populated with NaN from the current coordinate values.
        tmp_nd_coord_values = dataset[c_name].values
        # Create slices for the all grid dimensions that are also the coordinate dimensions.
        # For other dimension, select the first element (with index 0)
        slices = tuple(slice(None) if name in coord_dims else 0 for name in headers_dims)
        # Create a boolean mask for the live trace values with the dimensions of the coordinate
        # Q: Should we be using the trace_mask boolean array instead of grid map?
        not_null = grid_map_values[slices] != UINT32_MAX

        ch_reshaped = coord_headers_values.reshape(grid_map_values.shape)
        # Select a subset of the coordinate_headers that have unique values
        cr_reduced = ch_reshaped[slices]

        # Validate the all reduced dimensions had identical values
        axes_to_check = tuple(i for i, dim in enumerate(headers_dims) if dim not in coord_dims)
        coord_is_good[c_name] = _check_dimensions_values_identical(ch_reshaped, axes_to_check)

        # Save the unique coordinate values for live traces only
        tmp_nd_coord_values[not_null] = cr_reduced.ravel()
        dataset[c_name][:] = tmp_nd_coord_values

        drop_vars_delayed.append(c_name)
    return coord_is_good, dataset, drop_vars_delayed


def _get_horizontal_coordinate_unit(segy_headers: list[Dimension]) -> AllUnits | None:
    """Get the coordinate unit from the SEG-Y headers."""
    name = TraceHeaderFieldsRev0.COORDINATE_UNIT.name.upper()
    unit_hdr = next((c for c in segy_headers if c.name.upper() == name), None)
    if unit_hdr is None or len(unit_hdr.coords) == 0:
        # If the coordinate unit header is not found or empty, return None
        # This is a common case for SEG-Y files, where the coordinate unit is not specified
        return None

    if segy_MeasurementSystem(unit_hdr.coords[0]) == segy_MeasurementSystem.METERS:
        # If the coordinate unit is in meters, return "m"
        return AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))
    if segy_MeasurementSystem(unit_hdr.coords[0]) == segy_MeasurementSystem.FEET:
        # If the coordinate unit is in feet, return "ft"
        return AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT))
    err = f"Unsupported coordinate unit value: {unit_hdr.value[0]} in SEG-Y file."
    raise ValueError(err)


def _populate_coordinates(
    dataset: xr_Dataset,
    grid: Grid,
    coords: dict[str, SegyHeaderArray],
) -> tuple[xr_Dataset, list[str]]:
    """Populate dim and non-dim coordinates in the xarray dataset and write to Zarr.

    This will write the xr Dataset with coords and dimensions, but empty traces and headers.

    Args:
        dataset: The xarray dataset to populate.
        grid: The grid object containing the grid map.
        coords: The non-dim coordinates to populate.

    Returns:
        Xarray dataset with filled coordinates and updated variables to drop after writing
    """
    drop_vars_delayed = []
    # Populate the dimension coordinate variables (1-D arrays)
    dataset, drop_vars_delayed = populate_dim_coordinates(dataset, grid, drop_vars_delayed=drop_vars_delayed)

    # Populate the non-dimension coordinate variables (N-dim arrays)
    is_good, dataset, drop_vars_delayed = _populate_non_dim_coordinates(
        dataset, grid, coordinate_headers=coords, drop_vars_delayed=drop_vars_delayed
    )
    if not all(is_good.values()):
        failed_dims = [key for key, value in is_good.items() if not value]
        err = f"Non-dim coordinate(s) {failed_dims} have non-identical values " + \
            f"along reduced dimensions."
        raise ValueError(err)

    dataset = dataset.drop_vars(drop_vars_delayed)
    return dataset, drop_vars_delayed


def _add_segy_ingest_attributes(dataset: Dataset, segy_file: SegyFile, grid_overrides: dict[str, Any] | None) -> None:
    text_header = segy_file.text_header.splitlines()
    # Validate:
    # text_header this should be a 40-items array of strings with width of 80 characters.
    item_count = 40
    if len(text_header) != item_count:
        err = f"Invalid text header count: expected {item_count}, got {len(text_header)}"
        raise ValueError(err)
    char_count = 80
    for i, line in enumerate(text_header):
        if len(line) != char_count:
            err = f"Invalid text header {i} line length: expected {char_count}, got {len(line)}"
            raise ValueError(err)
    ext_text_header = segy_file.ext_text_header

    # If using SegyFile.ext_text_header this should be a minimum of 40 elements and must
    # capture all textual information (ensure text_header is a subset of ext_text_header).
    if ext_text_header is not None:
        for ext_hdr in ext_text_header:
            text_header.append(ext_hdr.splitlines())

    # Handle case where it may not have any metadata yet
    if dataset.metadata.attributes is None:
        dataset.metadata.attributes = {}

    segy_attributes = {
        "textHeader": text_header,
        "binaryHeader": segy_file.binary_header.to_dict(),
    }

    if grid_overrides is not None:
        segy_attributes["gridOverrides"] = grid_overrides

    # Update the attributes with the text and binary headers.
    dataset.metadata.attributes.update(segy_attributes)


def segy_to_mdio(  # noqa PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: dict[str, Any] | None = None,
) -> None:
    """A function that converts a SEG-Y file to an MDIO v1 file.

    Ingest a SEG-Y file according to the segy_spec. This could be a spec from registry or custom.

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        mdio_template: The MDIO template to use for the conversion.
        input_path: The universal path of the input SEG-Y file.
        output_path: The universal path for the output MDIO v1 file.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.
        grid_overrides: Option to add grid overrides.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
    """
    input_path = _normalize_path(input_path)
    output_path = _normalize_path(output_path)

    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    segy_settings = SegySettings(storage_options=input_path.storage_options)
    segy_file = SegyFile(url=input_path.as_posix(), spec=segy_spec, settings=segy_settings)

    segy_dimensions, segy_headers = _scan_for_headers(segy_file, mdio_template, grid_overrides)

    grid = _build_and_check_grid(segy_dimensions, segy_file, segy_headers)

    _, non_dim_coords = _get_coordinates(grid, segy_headers, mdio_template)
    # TODO(Altay): Turn this dtype into packed representation
    # https://github.com/TGSAI/mdio-python/issues/601
    headers = to_structured_type(segy_spec.trace.header.dtype)

    horizontal_unit = _get_horizontal_coordinate_unit(segy_dimensions)
    mdio_ds: Dataset = mdio_template.build_dataset(
        name=mdio_template.name,
        sizes=grid.shape,
        horizontal_coord_unit=horizontal_unit,
        headers=headers,
    )

    _add_segy_ingest_attributes(dataset=mdio_ds, segy_file=segy_file, grid_overrides=grid_overrides)

    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    xr_dataset, drop_vars_delayed = _populate_coordinates(
        dataset=xr_dataset,
        grid=grid,
        coords=non_dim_coords,
    )

    xr_dataset.trace_mask.data[:] = grid.live_mask

    # IMPORTANT: Do not drop the "trace_mask" here, as it will be used later in
    # blocked_io.to_zarr() -> _workers.trace_worker()

    # This will create the Zarr store with the correct structure but with empty arrays
    to_mdio(xr_dataset, output_path=output_path, mode="w", compute=False)

    # This will write the non-dimension coordinates and trace mask
    meta_ds = xr_dataset[drop_vars_delayed + ["trace_mask"]]
    to_mdio(meta_ds, output_path=output_path, mode="r+", compute=True)

    # Now we can drop them to simplify chunked write of the data variable
    xr_dataset = xr_dataset.drop_vars(drop_vars_delayed)

    # Write the headers and traces in chunks using grid_map to indicate dead traces
    default_variable_name = mdio_template.default_variable_name
    # This is an memory-expensive and time-consuming read-write operation
    # performed in chunks to save the memory
    blocked_io.to_zarr(
        segy_file=segy_file,
        output_path=output_path,
        grid_map=grid.map,
        dataset=xr_dataset,
        data_variable_name=default_variable_name,
    )
