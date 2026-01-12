"""Conversion from SEG-Y to MDIO v1 format."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

import numpy as np
import zarr
from segy.config import SegyFileSettings
from segy.config import SegyHeaderOverrides
from segy.standards.codes import MeasurementSystem as SegyMeasurementSystem
from segy.standards.fields import binary as binary_header_fields

from mdio.api.io import _normalize_path
from mdio.api.io import to_mdio
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.units import AngleUnitEnum
from mdio.builder.schemas.v1.units import AngleUnitModel
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.constants import ZarrFormat
from mdio.converters.exceptions import GridTraceCountError
from mdio.converters.exceptions import GridTraceSparsityError
from mdio.converters.type_converter import to_structured_type
from mdio.core.config import MDIOSettings
from mdio.core.grid import Grid
from mdio.core.utils_write import MAX_COORDINATES_BYTES
from mdio.core.utils_write import MAX_SIZE_LIVE_MASK
from mdio.core.utils_write import get_constrained_chunksize
from mdio.segy import blocked_io
from mdio.segy.file import get_segy_file_info
from mdio.segy.scalar import SCALE_COORDINATE_KEYS
from mdio.segy.scalar import _apply_coordinate_scalar
from mdio.segy.utilities import get_grid_plan

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from segy.arrays import HeaderArray as SegyHeaderArray
    from segy.schema import SegySpec
    from upath import UPath
    from xarray import Dataset as xr_Dataset

    from mdio.builder.schemas import Dataset
    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.core.dimension import Dimension
    from mdio.segy.file import SegyFileArguments
    from mdio.segy.file import SegyFileInfo

logger = logging.getLogger(__name__)


MEASUREMENT_SYSTEM_KEY = binary_header_fields.Rev0.MEASUREMENT_SYSTEM_CODE.model.name
ANGLE_UNIT_KEYS = ["angle", "azimuth"]
SPATIAL_UNIT_KEYS = [
    "cdp_x",
    "cdp_y",
    "source_coord_x",
    "source_coord_y",
    "group_coord_x",
    "group_coord_y",
    "offset",
]


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
    """
    settings = MDIOSettings()
    # Calculate total possible traces in the grid (excluding sample dimension)
    grid_traces = np.prod(grid.shape[:-1], dtype=np.uint64)

    # Handle division by zero if num_traces is 0
    sparsity_ratio = float("inf") if num_traces == 0 else grid_traces / num_traces

    # Fetch and validate environment variables
    warning_ratio = settings.grid_sparsity_ratio_warn
    error_ratio = settings.grid_sparsity_ratio_limit
    ignore_checks = settings.ignore_checks

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


def _patch_add_coordinates_for_non_binned(
    template: AbstractDatasetTemplate,
    non_binned_dims: set[str],
) -> None:
    """Patch template's _add_coordinates to skip adding non-binned dims as dimension coordinates.

    When NonBinned override is used, dimensions like 'offset' or 'azimuth' become coordinates
    instead of dimensions. However, template subclasses may still try to add them as 1D
    dimension coordinates (e.g., with dimensions=("offset",)). Since 'offset' is no longer
    a dimension, the builder substitutes 'trace', resulting in wrong coordinate dimensions.

    This function patches the template's _add_coordinates method to intercept calls to
    builder.add_coordinate and skip adding coordinates that are non-binned dims with
    single-element dimension tuples. These coordinates will be added later by build_dataset
    with the correct spatial_dimension_names (e.g., (inline, crossline, trace)).

    Args:
        template: The template to patch
        non_binned_dims: Set of dimension names that became coordinates due to NonBinned override
    """
    # Check if already patched to avoid duplicate patching
    if hasattr(template, "_mdio_non_binned_patched"):
        return

    # Store the original _add_coordinates method
    original_add_coordinates = template._add_coordinates

    def patched_add_coordinates() -> None:
        """Wrapper that intercepts builder.add_coordinate calls for non-binned dims."""
        # Store the original add_coordinate method from the builder
        original_builder_add_coordinate = template._builder.add_coordinate

        def filtered_add_coordinate(  # noqa: ANN202
            name: str,
            *,
            dimensions: tuple[str, ...],
            **kwargs,  # noqa: ANN003
        ):
            """Skip adding non-binned dims as 1D dimension coordinates."""
            # Check if this is a non-binned dim being added as a 1D dimension coordinate
            # (i.e., the coordinate name matches a non-binned dim and has only 1 dimension)
            if name in non_binned_dims and len(dimensions) == 1:
                logger.debug(
                    "Skipping 1D coordinate '%s' with dims %s - will be added with full spatial dims",
                    name,
                    dimensions,
                )
                return template._builder  # Return builder for chaining, but don't add

            # Otherwise, call the original method
            return original_builder_add_coordinate(name, dimensions=dimensions, **kwargs)

        # Temporarily replace builder's add_coordinate
        template._builder.add_coordinate = filtered_add_coordinate

        try:
            # Call the original _add_coordinates
            original_add_coordinates()
        finally:
            # Restore the original add_coordinate method
            template._builder.add_coordinate = original_builder_add_coordinate

    # Replace the template's _add_coordinates method
    template._add_coordinates = patched_add_coordinates

    # Mark as patched to prevent duplicate patching
    template._mdio_non_binned_patched = True


def _update_template_from_grid_overrides(
    template: AbstractDatasetTemplate,
    grid_overrides: dict[str, Any] | None,
    segy_dimensions: list[Dimension],
    full_chunk_shape: tuple[int, ...],
    chunk_size: tuple[int, ...],
) -> None:
    """Update template attributes to match grid plan results after grid overrides.

    This function modifies the template in-place to reflect changes from grid overrides:
    - Updates chunk shape if it changed due to overrides
    - Updates dimension names if they changed due to overrides
    - Adds non-binned dimensions as coordinates for NonBinned override
    - Patches _add_coordinates to skip adding non-binned dims as dimension coordinates

    Args:
        template: The template to update
        grid_overrides: Grid override configuration
        segy_dimensions: Dimensions returned from grid planning
        full_chunk_shape: Original template chunk shape
        chunk_size: Chunk size returned from grid planning
    """
    # Update template to match grid_plan results after grid overrides
    # Extract actual spatial dimensions from segy_dimensions (excluding vertical dimension)
    actual_spatial_dims = tuple(dim.name for dim in segy_dimensions[:-1])

    # Align chunk_size with actual dimensions - truncate if dimensions were filtered out
    num_actual_spatial = len(actual_spatial_dims)
    num_chunk_spatial = len(chunk_size) - 1  # Exclude vertical dimension chunk
    if num_actual_spatial != num_chunk_spatial:
        # Truncate chunk_size to match actual dimensions
        chunk_size = chunk_size[:num_actual_spatial] + (chunk_size[-1],)

    if full_chunk_shape != chunk_size:
        logger.debug(
            "Adjusting template chunk shape from %s to %s to match grid after overrides",
            full_chunk_shape,
            chunk_size,
        )
        template._var_chunk_shape = chunk_size

    # Update dimensions if they don't match grid_plan results
    if template.spatial_dimension_names != actual_spatial_dims:
        logger.debug(
            "Adjusting template dimensions from %s to %s to match grid after overrides",
            template.spatial_dimension_names,
            actual_spatial_dims,
        )
        template._dim_names = actual_spatial_dims + (template.trace_domain,)

    # If using NonBinned override, expose non-binned dims as logical coordinates on the template instance
    # and patch _add_coordinates to skip adding them as 1D dimension coordinates
    if grid_overrides and "NonBinned" in grid_overrides and "non_binned_dims" in grid_overrides:
        non_binned_dims = tuple(grid_overrides["non_binned_dims"])
        if non_binned_dims:
            logger.debug(
                "NonBinned grid override: exposing non-binned dims as coordinates: %s",
                non_binned_dims,
            )
            # Append any missing names; keep existing order and avoid duplicates
            existing = set(template.coordinate_names)
            to_add = tuple(n for n in non_binned_dims if n not in existing)
            if to_add:
                template._logical_coord_names = template._logical_coord_names + to_add

            # Patch _add_coordinates to skip adding non-binned dims as 1D dimension coordinates
            # This prevents them from being added with wrong dimensions (e.g., just "trace")
            # They will be added later by build_dataset with full spatial_dimension_names
            _patch_add_coordinates_for_non_binned(template, set(non_binned_dims))


def _scan_for_headers(
    segy_file_kwargs: SegyFileArguments,
    segy_file_info: SegyFileInfo,
    template: AbstractDatasetTemplate,
    grid_overrides: dict[str, Any] | None = None,
) -> tuple[list[Dimension], SegyHeaderArray]:
    """Extract trace dimensions and index headers from the SEG-Y file.

    This is an expensive operation.
    It scans the SEG-Y file in chunks by using ProcessPoolExecutor.

    Note:
        If grid_overrides are applied to the template before calling this function,
        the chunk_size returned from get_grid_plan should match the template's chunk shape.
    """
    full_chunk_shape = template.full_chunk_shape
    segy_dimensions, chunk_size, segy_headers = get_grid_plan(
        segy_file_kwargs=segy_file_kwargs,
        segy_file_info=segy_file_info,
        return_headers=True,
        template=template,
        chunksize=full_chunk_shape,
        grid_overrides=grid_overrides,
    )

    _update_template_from_grid_overrides(
        template=template,
        grid_overrides=grid_overrides,
        segy_dimensions=segy_dimensions,
        full_chunk_shape=full_chunk_shape,
        chunk_size=chunk_size,
    )

    return segy_dimensions, segy_headers


def _build_and_check_grid(
    segy_dimensions: list[Dimension],
    segy_file_info: SegyFileInfo,
    segy_headers: SegyHeaderArray,
) -> Grid:
    """Build and check the grid from the SEG-Y headers and dimensions.

    Args:
        segy_dimensions: List of of all SEG-Y dimensions to build grid from.
        segy_file_info: SegyFileInfo instance containing the SEG-Y file information.
        segy_headers: Headers read in from SEG-Y file for building the trace map.

    Returns:
        A grid instance populated with the dimensions and trace index map.

    Raises:
        GridTraceCountError: If number of traces in SEG-Y file does not match the parsed grid
    """
    grid = Grid(dims=segy_dimensions)
    num_traces = segy_file_info.num_traces
    grid_density_qc(grid, num_traces)
    grid.build_map(segy_headers)
    # Check grid validity by comparing trace numbers
    if np.sum(grid.live_mask) != num_traces:
        for dim_name in grid.dim_names:
            dim_min, dim_max = grid.get_min(dim_name), grid.get_max(dim_name)
            logger.warning("%s min: %s max: %s", dim_name, dim_min, dim_max)
        logger.warning("Ingestion grid shape: %s.", grid.shape)
        raise GridTraceCountError(np.sum(grid.live_mask), num_traces)
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
        # Copy the data to allow segy_headers to be garbage collected
        non_dim_coords[coord_name] = np.array(segy_headers[coord_name])

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
    spatial_coordinate_scalar: int,
) -> tuple[xr_Dataset, list[str]]:
    """Populate the xarray dataset with coordinate variables.

    Memory optimization: Processes coordinates one at a time and explicitly
    releases intermediate arrays to reduce peak memory usage.
    """
    non_data_domain_dims = grid.dim_names[:-1]  # minus the data domain dimension

    # Process coordinates one at a time to minimize peak memory
    coord_names = list(coordinates.keys())
    for coord_name in coord_names:
        coord_values = coordinates.pop(coord_name)  # Remove from dict to free memory
        da_coord = dataset[coord_name]

        # Get coordinate shape from dataset (uses dask shape, no memory allocation)
        coord_shape = da_coord.shape

        # Create output array with fill value
        fill_value = da_coord.encoding.get("_FillValue") or da_coord.encoding.get("fill_value")
        if fill_value is None:
            fill_value = np.nan
        tmp_coord_values = np.full(coord_shape, fill_value, dtype=da_coord.dtype)

        # Compute slices for this coordinate's dimensions
        coord_axes = tuple(non_data_domain_dims.index(coord_dim) for coord_dim in da_coord.dims)
        coord_slices = tuple(slice(None) if idx in coord_axes else 0 for idx in range(len(non_data_domain_dims)))

        # Read only the required slice from grid map
        coord_trace_indices = np.asarray(grid.map[coord_slices])

        # Find valid (non-null) indices
        not_null = coord_trace_indices != grid.map.fill_value

        # Populate values efficiently
        if not_null.any():
            valid_indices = coord_trace_indices[not_null]
            tmp_coord_values[not_null] = coord_values[valid_indices]

        # Apply scalar if needed
        if coord_name in SCALE_COORDINATE_KEYS:
            tmp_coord_values = _apply_coordinate_scalar(tmp_coord_values, spatial_coordinate_scalar)

        # Assign to dataset
        dataset[coord_name][:] = tmp_coord_values
        drop_vars_delayed.append(coord_name)

        # Explicitly release intermediate arrays
        del tmp_coord_values, coord_trace_indices, not_null, coord_values

        # TODO(Altay): Add verification of reduced coordinates being the same as the first
        # https://github.com/TGSAI/mdio-python/issues/645

    return dataset, drop_vars_delayed


def _get_spatial_coordinate_unit(segy_file_info: SegyFileInfo) -> LengthUnitModel | None:
    """Get the coordinate unit from the SEG-Y headers."""
    measurement_system_code = int(segy_file_info.binary_header_dict[MEASUREMENT_SYSTEM_KEY])

    if measurement_system_code not in (1, 2):
        logger.warning(
            "Unexpected value in coordinate unit (%s) header: %s. Can't extract coordinate unit and will "
            "ingest without coordinate units.",
            MEASUREMENT_SYSTEM_KEY,
            measurement_system_code,
        )
        return None

    if measurement_system_code == SegyMeasurementSystem.METERS:
        unit = LengthUnitEnum.METER
    if measurement_system_code == SegyMeasurementSystem.FEET:
        unit = LengthUnitEnum.FOOT

    return LengthUnitModel(length=unit)


def _update_template_units(template: AbstractDatasetTemplate, unit: LengthUnitModel | None) -> AbstractDatasetTemplate:
    """Update the template with dynamic and some pre-defined units."""
    # Add units for pre-defined: angle and azimuth etc.
    new_units = {key: AngleUnitModel(angle=AngleUnitEnum.DEGREES) for key in ANGLE_UNIT_KEYS}

    # If a spatial unit is not provided, we return as is
    if unit is None:
        template.add_units(new_units)
        return template

    # Dynamically add units based on the spatial coordinate unit
    for key in SPATIAL_UNIT_KEYS:
        current_value = template.get_unit_by_key(key)
        if current_value is not None:
            logger.warning("Unit for %s already in template. Will keep the original unit: %s", key, current_value)
            continue

        new_units[key] = unit

    template.add_units(new_units)
    return template


def _populate_coordinates(
    dataset: xr_Dataset,
    grid: Grid,
    coords: dict[str, SegyHeaderArray],
    spatial_coordinate_scalar: int,
) -> tuple[xr_Dataset, list[str]]:
    """Populate dim and non-dim coordinates in the xarray dataset and write to Zarr.

    This will write the xr Dataset with coords and dimensions, but empty traces and headers.

    Args:
        dataset: The xarray dataset to populate.
        grid: The grid object containing the grid map.
        coords: The non-dim coordinates to populate.
        spatial_coordinate_scalar: The X/Y coordinate scalar from the SEG-Y file.

    Returns:
        Xarray dataset with filled coordinates and updated variables to drop after writing
    """
    drop_vars_delayed = []
    # Populate the dimension coordinate variables (1-D arrays)
    dataset, drop_vars_delayed = populate_dim_coordinates(dataset, grid, drop_vars_delayed=drop_vars_delayed)

    # Populate the non-dimension coordinate variables (N-dim arrays)
    dataset, drop_vars_delayed = populate_non_dim_coordinates(
        dataset,
        grid,
        coordinates=coords,
        drop_vars_delayed=drop_vars_delayed,
        spatial_coordinate_scalar=spatial_coordinate_scalar,
    )

    return dataset, drop_vars_delayed


def _add_segy_file_headers(xr_dataset: xr_Dataset, segy_file_info: SegyFileInfo) -> xr_Dataset:
    settings = MDIOSettings()

    if not settings.save_segy_file_header:
        return xr_dataset

    expected_rows = 40
    expected_cols = 80

    text_header_rows = segy_file_info.text_header.splitlines()
    text_header_cols_bad = [len(row) != expected_cols for row in text_header_rows]

    if len(text_header_rows) != expected_rows:
        err = f"Invalid text header count: expected {expected_rows}, got {len(segy_file_info.text_header)}"
        raise ValueError(err)

    if any(text_header_cols_bad):
        err = f"Invalid text header columns: expected {expected_cols} per line."
        raise ValueError(err)

    xr_dataset["segy_file_header"] = ((), "")
    xr_dataset["segy_file_header"].attrs.update(
        {
            "textHeader": segy_file_info.text_header,
            "binaryHeader": segy_file_info.binary_header_dict,
        }
    )
    if settings.raw_headers:
        raw_binary_base64 = base64.b64encode(segy_file_info.raw_binary_headers).decode("ascii")
        xr_dataset["segy_file_header"].attrs.update({"rawBinaryHeader": raw_binary_base64})

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
        chunk_shape = mdio_template.full_chunk_shape[:-1]

        # Create chunk grid metadata
        chunk_metadata = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=chunk_shape))

        # Add the raw headers variable using the builder's add_variable method
        mdio_template._builder.add_variable(
            name="raw_headers",
            long_name="Raw Headers",
            dimensions=mdio_template.spatial_dimension_names,
            data_type=ScalarType.BYTES240,
            compressor=Blosc(cname=BloscCname.zstd),
            coordinates=None,  # No coordinates as specified
            metadata=VariableMetadata(chunk_grid=chunk_metadata),
        )

    # Replace the template's _add_variables method
    mdio_template._add_variables = enhanced_add_variables

    # Mark the template as enhanced to prevent duplicate monkey-patching
    mdio_template._mdio_raw_headers_enhanced = True

    return mdio_template


def _chunk_variable(ds: Dataset, target_variable_name: str) -> None:
    """Determines and sets the chunking for a specific Variable in the Dataset."""
    # Find variable index by name
    index = next((i for i, obj in enumerate(ds.variables) if obj.name == target_variable_name), None)

    def determine_target_size(var_type: str) -> int:
        """Determines the target size (in bytes) for a Variable based on its type."""
        if var_type == "bool":
            return MAX_SIZE_LIVE_MASK
        return MAX_COORDINATES_BYTES

    # Create the chunk grid metadata
    var_type = ds.variables[index].data_type
    full_shape = tuple(dim.size for dim in ds.variables[index].dimensions)
    target_size = determine_target_size(var_type)

    chunk_shape = get_constrained_chunksize(full_shape, var_type, target_size)
    chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=chunk_shape))

    # Create variable metadata if it doesn't exist
    if ds.variables[index].metadata is None:
        ds.variables[index].metadata = VariableMetadata()

    ds.variables[index].metadata.chunk_grid = chunk_grid


def _validate_spec_in_template(segy_spec: SegySpec, mdio_template: AbstractDatasetTemplate) -> None:
    """Validate that the SegySpec has all required fields in the MDIO template."""
    header_fields = {field.name for field in segy_spec.trace.header.fields}

    required_fields = set(mdio_template.spatial_dimension_names) | set(mdio_template.coordinate_names)
    required_fields = required_fields - set(mdio_template.calculated_dimension_names)  # remove to be calculated
    required_fields = required_fields | {"coordinate_scalar"}  # ensure coordinate scalar is always present
    missing_fields = required_fields - header_fields

    if missing_fields:
        err = (
            f"Required fields {sorted(missing_fields)} for template {mdio_template.name} "
            f"not found in the provided segy_spec"
        )
        raise ValueError(err)


def segy_to_mdio(  # noqa PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: dict[str, Any] | None = None,
    segy_header_overrides: SegyHeaderOverrides | None = None,
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
        segy_header_overrides: Option to override specific SEG-Y headers during ingestion.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
    """
    settings = MDIOSettings()

    _validate_spec_in_template(segy_spec, mdio_template)

    input_path = _normalize_path(input_path)
    output_path = _normalize_path(output_path)

    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    segy_settings = SegyFileSettings(storage_options=input_path.storage_options)
    segy_file_kwargs: SegyFileArguments = {
        "url": input_path.as_posix(),
        "spec": segy_spec,
        "settings": segy_settings,
        "header_overrides": segy_header_overrides,
    }
    segy_file_info = get_segy_file_info(segy_file_kwargs)

    segy_dimensions, segy_headers = _scan_for_headers(
        segy_file_kwargs,
        segy_file_info,
        template=mdio_template,
        grid_overrides=grid_overrides,
    )
    grid = _build_and_check_grid(segy_dimensions, segy_file_info, segy_headers)

    _, non_dim_coords = _get_coordinates(grid, segy_headers, mdio_template)

    # Explicitly delete segy_headers to free memory - coordinate values have been copied
    del segy_headers

    header_dtype = to_structured_type(segy_spec.trace.header.dtype)

    if settings.raw_headers:
        if zarr.config.get("default_zarr_format") == ZarrFormat.V2:
            logger.warning("Raw headers are only supported for Zarr v3. Skipping raw headers.")
        else:
            logger.warning("MDIO__IMPORT__RAW_HEADERS is experimental and expected to change or be removed.")
            mdio_template = _add_raw_headers_to_template(mdio_template)

    spatial_unit = _get_spatial_coordinate_unit(segy_file_info)
    mdio_template = _update_template_units(mdio_template, spatial_unit)
    mdio_ds: Dataset = mdio_template.build_dataset(name=mdio_template.name, sizes=grid.shape, header_dtype=header_dtype)

    _add_grid_override_to_metadata(dataset=mdio_ds, grid_overrides=grid_overrides)

    # Dynamically chunk the variables based on their type
    _chunk_variable(ds=mdio_ds, target_variable_name="trace_mask")  # trace_mask is a Variable and not a Coordinate
    for coord in mdio_template.coordinate_names:
        _chunk_variable(ds=mdio_ds, target_variable_name=coord)

    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    xr_dataset, drop_vars_delayed = _populate_coordinates(
        dataset=xr_dataset,
        grid=grid,
        coords=non_dim_coords,
        spatial_coordinate_scalar=segy_file_info.coordinate_scalar,
    )

    xr_dataset = _add_segy_file_headers(xr_dataset, segy_file_info)

    xr_dataset.trace_mask.data[:] = grid.live_mask
    # IMPORTANT: Do not drop the "trace_mask" here, as it will be used later in
    # blocked_io.to_zarr() -> _workers.trace_worker()

    # This will create the Zarr store with the correct structure but with empty arrays
    to_mdio(xr_dataset, output_path=output_path, mode="w", compute=False)

    # This will write the non-dimension coordinates and trace mask
    # We also remove dimensions that don't have associated coordinates
    unindexed_dims = [d for d in xr_dataset.dims if d not in xr_dataset.coords]
    [drop_vars_delayed.remove(d) for d in unindexed_dims]
    meta_ds = xr_dataset[drop_vars_delayed + ["trace_mask"]]
    to_mdio(meta_ds, output_path=output_path, mode="r+", compute=True)

    # Now we can drop them to simplify chunked write of the data variable
    xr_dataset = xr_dataset.drop_vars(drop_vars_delayed)

    # Write the headers and traces in chunks using grid_map to indicate dead traces
    default_variable_name = mdio_template.default_variable_name
    # This is an memory-expensive and time-consuming read-write operation
    # performed in chunks to save the memory
    blocked_io.to_zarr(
        segy_file_kwargs=segy_file_kwargs,
        output_path=output_path,
        grid_map=grid.map,
        dataset=xr_dataset,
        data_variable_name=default_variable_name,
    )
