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
    from segy.arrays import HeaderArray as SegyHeaderArray
    from segy.schema import SegySpec
    from xarray import Dataset as xr_Dataset

    from mdio.core.dimension import Dimension
    from mdio.core.storage_location import StorageLocation
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
    segy_file: SegyFile, template: AbstractDatasetTemplate
) -> tuple[list[Dimension], SegyHeaderArray]:
    """Extract trace dimensions and index headers from the SEG-Y file.

    This is an expensive operation.
    It scans the SEG-Y file in chunks by using ProcessPoolExecutor
    """
    # TODO(Dmitriy): Enhance get_grid_plan_v1 to return only needed headers
    # https://github.com/TGSAI/mdio-python/issues/589
    # TODO(Dmitriy): implement grid overrides
    # https://github.com/TGSAI/mdio-python/issues/585
    # The 'grid_chunksize' is used only for grid_overrides
    # While we do not support grid override, we can set it to None
    grid_chunksize = None
    segy_dimensions, chunksize, segy_headers = get_grid_plan(
        segy_file=segy_file,
        return_headers=True,
        template=template,
        chunksize=grid_chunksize,
        grid_overrides=None,
    )
    return segy_dimensions, segy_headers


def _get_coordinates(
    segy_dimensions: list[Dimension],
    segy_headers: SegyHeaderArray,
    mdio_template: AbstractDatasetTemplate,
) -> tuple[list[Dimension], dict[str, SegyHeaderArray]]:
    """Get the data dim and non-dim coordinates from the SEG-Y headers and MDIO template.

    Select a subset of the segy_dimensions that corresponds to the MDIO dimensions
    The dimensions are ordered as in the MDIO template.
    The last dimension is always the vertical domain dimension

    Args:
        segy_dimensions: List of of all SEG-Y dimensions.
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
    dim_names = [dim.name for dim in segy_dimensions]
    for dim_name in mdio_template.dimension_names:
        try:
            dim_index = dim_names.index(dim_name)
        except ValueError:
            err = f"Dimension '{dim_name}' was not found in SEG-Y dimensions."
            raise ValueError(err) from err
        dimensions_coords.append(segy_dimensions[dim_index])

    non_dim_coords: dict[str, SegyHeaderArray] = {}
    available_headers = segy_headers.dtype.names
    for coord_name in mdio_template.coordinate_names:
        if coord_name not in available_headers:
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


def populate_non_dim_coordinates(
    dataset: xr_Dataset,
    grid: Grid,
    coordinates: dict[str, SegyHeaderArray],
    drop_vars_delayed: list[str],
) -> tuple[xr_Dataset, list[str]]:
    """Populate the xarray dataset with coordinate variables."""
    not_null = grid.map[:] != UINT32_MAX
    for c_name, c_values in coordinates.items():
        dataset[c_name].values[not_null] = c_values
        drop_vars_delayed.append(c_name)
    return dataset, drop_vars_delayed


def _get_horizontal_coordinate_unit(segy_headers: list[Dimension]) -> LengthUnitEnum | None:
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
    dataset, vars_to_drop_later = populate_dim_coordinates(
        dataset, grid, drop_vars_delayed=drop_vars_delayed
    )

    # Populate the non-dimension coordinate variables (N-dim arrays)
    dataset, vars_to_drop_later = populate_non_dim_coordinates(
        dataset, grid, coordinates=coords, drop_vars_delayed=drop_vars_delayed
    )

    return dataset, drop_vars_delayed


def segy_to_mdio(
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_location: StorageLocation,
    output_location: StorageLocation,
    overwrite: bool = False,
) -> None:
    """A function that converts a SEG-Y file to an MDIO v1 file.

    Ingest a SEG-Y file according to the segy_spec. This could be a spec from registry or custom.

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        mdio_template: The MDIO template to use for the conversion.
        input_location: The storage location of the input SEG-Y file.
        output_location: The storage location for the output MDIO v1 file.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
        GridTraceCountError: If number of traces in SEG-Y file does not match the parsed grid
    """
    if not overwrite and output_location.exists():
        err = f"Output location '{output_location.uri}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    segy_settings = SegySettings(storage_options=input_location.options)
    segy_file = SegyFile(url=input_location.uri, spec=segy_spec, settings=segy_settings)

    # Scan the SEG-Y file for headers
    segy_dimensions, segy_headers = _scan_for_headers(segy_file, mdio_template)

    # Build grid
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

    dimensions, non_dim_coords = _get_coordinates(segy_dimensions, segy_headers, mdio_template)
    shape = [len(dim.coords) for dim in dimensions]
    headers = to_structured_type(segy_headers.dtype)

    horizontal_unit = _get_horizontal_coordinate_unit(segy_dimensions)
    mdio_ds: Dataset = mdio_template.build_dataset(
        name=mdio_template.name, sizes=shape, horizontal_coord_unit=horizontal_unit, headers=headers
    )

    # TODO(Dmitriy Repin): work around of the bug
    # https://github.com/TGSAI/mdio-python/issues/582
    # Do not set _FillValue for the "header" variable, which has structured data type
    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds, no_fill_var_names=["headers"])

    xr_dataset, drop_vars_delayed = _populate_coordinates(
        dataset=xr_dataset,
        grid=grid,
        coords=non_dim_coords,
    )

    xr_dataset.trace_mask.data[:] = grid.live_mask

    # TODO(Dmitriy Repin): Write out text and binary headers.
    # https://github.com/TGSAI/mdio-python/issues/595

    # IMPORTANT: Do not drop the "trace_mask" here, as it will be used later in
    # blocked_io.to_zarr_v1() -> _workers.trace_worker_v1()

    # Write the xarray dataset to Zarr with as following:
    # Populated arrays:
    # - 1D dimensional coordinates
    # - ND non-dimensional coordinates
    # - ND trace_mask
    # Empty arrays (will be populated later in chunks):
    # - ND+1 traces
    # - ND headers (no _FillValue set due to the bug https://github.com/TGSAI/mdio-python/issues/582)
    # This will create the Zarr store with the correct structure
    # TODO(Dmitriy Repin): do chunked write for non-dimensional coordinates and trace_mask
    # https://github.com/TGSAI/mdio-python/issues/587
    xr_dataset.to_zarr(
        store=output_location.uri, mode="w", write_empty_chunks=False, zarr_format=2, compute=True
    )

    # Now we can drop them to simplify chunked write of the data variable
    xr_dataset = xr_dataset.drop_vars(drop_vars_delayed)

    # Write the headers and traces in chunks using grid_map to indicate dead traces
    data_variable_name = mdio_template.trace_variable_name
    # This is an memory-expensive and time-consuming read-write operation
    # performed in chunks to save the memory
    blocked_io.to_zarr_v1(
        segy_file=segy_file,
        output_location=output_location,
        grid_map=grid.map,
        dataset=xr_dataset,
        data_variable_name=data_variable_name,
    )
