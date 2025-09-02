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


def populate_non_dim_coordinates(
    dataset: xr_Dataset,
    grid: Grid,
    coordinates: dict[str, SegyHeaderArray],
    drop_vars_delayed: list[str],
) -> tuple[xr_Dataset, list[str]]:
    """Populate the xarray dataset with coordinate variables."""
    not_null = grid.map[:] != UINT32_MAX
    for c_name, c_values in coordinates.items():
        c_tmp_array = dataset[c_name].values
        c_tmp_array[not_null] = c_values
        dataset[c_name][:] = c_tmp_array
        drop_vars_delayed.append(c_name)
    return dataset, drop_vars_delayed


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
    dataset, vars_to_drop_later = populate_dim_coordinates(dataset, grid, drop_vars_delayed=drop_vars_delayed)

    # Populate the non-dimension coordinate variables (N-dim arrays)
    dataset, vars_to_drop_later = populate_non_dim_coordinates(
        dataset, grid, coordinates=coords, drop_vars_delayed=drop_vars_delayed
    )

    return dataset, drop_vars_delayed


def _add_text_binary_headers(dataset: Dataset, segy_file: SegyFile) -> None:
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
        dataset.attrs["attributes"] = {}

    # Update the attributes with the text and binary headers.
    dataset.metadata.attributes.update(
        {
            "textHeader": text_header,
            "binaryHeader": segy_file.binary_header.to_dict(),
        }
    )

def _chunk_variable(ds: Dataset, grid: Grid, variable_name: str) -> None:
    from mdio.schemas.v1.dataset_builder import ChunkGridMetadata
    from mdio.schemas.metadata import ChunkGridMetadata
    from mdio.schemas.chunk_grid import RegularChunkGrid, RegularChunkShape
    from mdio.core.utils_write import get_constrained_chunksize
    from mdio.core.utils_write import MAX_SIZE_LIVE_MASK
    from mdio.schemas.v1.variable import VariableMetadata
    
    # Find the variable by name
    idx = -1
    for i in range(len(ds.variables)):
        if ds.variables[i].name == variable_name:
            idx = i
            break
    if idx == -1:
        raise ValueError(f"Variable '{variable_name}' not found in dataset.")
    
    # Create the chunk grid metadata
    t = ds.variables[idx].data_type
    if t == "bool":
        target_size = MAX_SIZE_LIVE_MASK
    else:
        target_size = 128*1024**2

    chunks = ChunkGridMetadata(chunk_grid=RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=get_constrained_chunksize(grid.live_mask.shape, t, target_size))))

    # Update the variable's metadata
    if ds.variables[idx].metadata is None:
        # Create new metadata with the chunk grid
        ds.variables[idx].metadata = VariableMetadata(chunk_grid=chunks.chunk_grid)
    else:
        # Update existing metadata
        ds.variables[idx].metadata.chunk_grid = chunks.chunk_grid

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
    """
    if not overwrite and output_location.exists():
        err = f"Output location '{output_location.uri}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    segy_settings = SegySettings(storage_options=input_location.options)
    segy_file = SegyFile(url=input_location.uri, spec=segy_spec, settings=segy_settings)

    # Scan the SEG-Y file for headers
    segy_dimensions, segy_headers = _scan_for_headers(segy_file, mdio_template)

    grid = _build_and_check_grid(segy_dimensions, segy_file, segy_headers)

    dimensions, non_dim_coords = _get_coordinates(grid, segy_headers, mdio_template)
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

    _add_text_binary_headers(dataset=mdio_ds, segy_file=segy_file)
    _chunk_variable(ds=mdio_ds, grid=grid, variable_name="trace_mask")
    for coord in mdio_template.coordinate_names:
        _chunk_variable(ds=mdio_ds, grid=grid, variable_name=coord)

    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    xr_dataset, drop_vars_delayed = _populate_coordinates(
        dataset=xr_dataset,
        grid=grid,
        coords=non_dim_coords,
    )

    xr_dataset.trace_mask.data[:] = grid.live_mask

    # TODO(Dmitriy Repin): Write out text and binary headers.
    # https://github.com/TGSAI/mdio-python/issues/595

    # IMPORTANT: Do not drop the "trace_mask" here, as it will be used later in
    # blocked_io.to_zarr() -> _workers.trace_worker()

    # This will create the Zarr store with the correct structure but with empty arrays
    xr_dataset.to_zarr(store=output_location.uri, mode="w", write_empty_chunks=False, zarr_format=2, compute=False)

    # This will write the non-dimension coordinates and trace mask
    meta_ds = xr_dataset[drop_vars_delayed + ["trace_mask"]]
    meta_ds.to_zarr(store=output_location.uri, mode="r+", write_empty_chunks=False, zarr_format=2, compute=True)

    # Now we can drop them to simplify chunked write of the data variable
    xr_dataset = xr_dataset.drop_vars(drop_vars_delayed)

    # Write the headers and traces in chunks using grid_map to indicate dead traces
    data_variable_name = mdio_template.trace_variable_name
    # This is an memory-expensive and time-consuming read-write operation
    # performed in chunks to save the memory
    blocked_io.to_zarr(
        segy_file=segy_file,
        output_location=output_location,
        grid_map=grid.map,
        dataset=xr_dataset,
        data_variable_name=data_variable_name,
    )
