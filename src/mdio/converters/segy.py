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
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.converters.exceptions import EnvironmentFormatError
from mdio.converters.exceptions import GridTraceCountError
from mdio.converters.exceptions import GridTraceSparsityError
from mdio.converters.type_converter import to_structured_type
from mdio.core.grid import Grid
from mdio.segy import blocked_io
from mdio.segy.utilities import get_grid_plan

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from segy.arrays import HeaderArray as SegyHeaderArray
    from segy.schema import SegySpec
    from upath import UPath
    from xarray import Dataset as xr_Dataset

    from mdio.builder.schemas import Dataset
    from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate
    from mdio.core.dimension import Dimension

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


def populate_non_dim_coordinates(
    dataset: xr_Dataset,
    grid: Grid,
    coordinates: dict[str, SegyHeaderArray],
    drop_vars_delayed: list[str],
) -> tuple[xr_Dataset, list[str]]:
    """Populate the xarray dataset with coordinate variables."""
    non_data_domain_dims = grid.dim_names[:-1]  # minus the data domain dimension
    for coord_name, coord_values in coordinates.items():
        da_coord = dataset[coord_name]
        tmp_coord_values = dataset[coord_name].values

        coord_axes = tuple(non_data_domain_dims.index(coord_dim) for coord_dim in da_coord.dims)
        coord_slices = tuple(slice(None) if idx in coord_axes else 0 for idx in range(len(non_data_domain_dims)))
        coord_trace_indices = grid.map[coord_slices]

        not_null = coord_trace_indices != grid.map.fill_value
        tmp_coord_values[not_null] = coord_values[coord_trace_indices[not_null]]
        dataset[coord_name][:] = tmp_coord_values
        drop_vars_delayed.append(coord_name)

        # TODO(Altay): Add verification of reduced coordinates being the same as the first
        # https://github.com/TGSAI/mdio-python/issues/645

    return dataset, drop_vars_delayed


def _get_horizontal_coordinate_unit(segy_headers: list[Dimension]) -> LengthUnitModel | None:
    """Get the coordinate unit from the SEG-Y headers."""
    name = TraceHeaderFieldsRev0.COORDINATE_UNIT.name.upper()
    unit_hdr = next((c for c in segy_headers if c.name.upper() == name), None)
    if unit_hdr is None or len(unit_hdr.coords) == 0:
        return None

    if segy_MeasurementSystem(unit_hdr.coords[0]) == segy_MeasurementSystem.METERS:
        unit = LengthUnitEnum.METER
    if segy_MeasurementSystem(unit_hdr.coords[0]) == segy_MeasurementSystem.FEET:
        unit = LengthUnitEnum.FOOT

    return LengthUnitModel(length=unit)


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


def _add_segy_file_headers(xr_dataset: xr_Dataset, segy_file: SegyFile) -> xr_Dataset:
    save_file_header = os.getenv("MDIO__IMPORT__SAVE_SEGY_FILE_HEADER", "") in ("1", "true", "yes", "on")
    if not save_file_header:
        return xr_dataset

    expected_rows = 40
    expected_cols = 80

    text_header = segy_file.text_header
    text_header_rows = text_header.splitlines()
    text_header_cols_bad = [len(row) != expected_cols for row in text_header_rows]

    if len(text_header_rows) != expected_rows:
        err = f"Invalid text header count: expected {expected_rows}, got {len(text_header)}"
        raise ValueError(err)

    if any(text_header_cols_bad):
        err = f"Invalid text header columns: expected {expected_cols} per line."
        raise ValueError(err)

    xr_dataset["segy_file_header"] = ((), "")
    xr_dataset["segy_file_header"].attrs.update(
        {
            "textHeader": text_header,
            "binaryHeader": segy_file.binary_header.to_dict(),
        }
    )

    return xr_dataset


def _add_grid_override_to_metadata(dataset: Dataset, grid_overrides: dict[str, Any] | None) -> None:
    """Add grid override to Dataset metadata if needed."""
    if dataset.metadata.attributes is None:
        dataset.metadata.attributes = {}

    if grid_overrides is not None:
        dataset.metadata.attributes["gridOverrides"] = grid_overrides


def _add_raw_headers_to_template(mdio_template: AbstractDatasetTemplate) -> AbstractDatasetTemplate:
    """Add raw headers capability to the MDIO template by monkey-patching its _add_variables method.

    This function modifies the template's _add_variables method to also add a raw headers variable
    with the following characteristics:
    - Same rank as the Headers variable (all dimensions except vertical)
    - Name: "RawHeaders"
    - Type: ScalarType.HEADERS
    - No coordinates
    - zstd compressor
    - No additional metadata
    - Chunked the same as the Headers variable
    Args:
        mdio_template: The MDIO template to mutate
    Returns:
        The mutated MDIO template
    """
    # Check if raw headers enhancement has already been applied to avoid duplicate additions
    if hasattr(mdio_template, "_mdio_raw_headers_enhanced"):
        return mdio_template

    # Store the original _add_variables method
    original_add_variables = mdio_template._add_variables

    def enhanced_add_variables() -> None:
        # Call the original method first
        original_add_variables()

        # Now add the raw headers variable
        chunk_shape = mdio_template._var_chunk_shape[:-1]

        # Create chunk grid metadata
        chunk_metadata = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=chunk_shape))

        # Add the raw headers variable using the builder's add_variable method
        mdio_template._builder.add_variable(
            name="raw_headers",
            long_name="Raw Headers",
            dimensions=mdio_template._dim_names[:-1],  # All dimensions except vertical
            data_type=ScalarType.HEADERS_V3,
            compressor=Blosc(cname=BloscCname.zstd),
            coordinates=None,  # No coordinates as specified
            metadata=VariableMetadata(chunk_grid=chunk_metadata),
        )

    # Replace the template's _add_variables method
    mdio_template._add_variables = enhanced_add_variables

    # Mark the template as enhanced to prevent duplicate monkey-patching
    mdio_template._mdio_raw_headers_enhanced = True

    return mdio_template


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
    header_dtype = to_structured_type(segy_spec.trace.header.dtype)

    if os.getenv("MDIO__DO_RAW_HEADERS") == "1":
        logger.warning("MDIO__DO_RAW_HEADERS is experimental and expected to change or be removed.")
        mdio_template = _add_raw_headers_to_template(mdio_template)

    horizontal_unit = _get_horizontal_coordinate_unit(segy_dimensions)
    mdio_ds: Dataset = mdio_template.build_dataset(
        name=mdio_template.name,
        sizes=grid.shape,
        horizontal_coord_unit=horizontal_unit,
        header_dtype=header_dtype,
    )

    _add_grid_override_to_metadata(dataset=mdio_ds, grid_overrides=grid_overrides)

    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    xr_dataset, drop_vars_delayed = _populate_coordinates(
        dataset=xr_dataset,
        grid=grid,
        coords=non_dim_coords,
    )

    xr_dataset = _add_segy_file_headers(xr_dataset, segy_file)

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
