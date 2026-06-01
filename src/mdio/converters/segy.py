"""Conversion from SEG-Y to MDIO v1 format."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import zarr
from segy.config import SegyFileSettings
from segy.config import SegyHeaderOverrides

from mdio.api.io import _normalize_path
from mdio.api.io import to_mdio
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.constants import ZarrFormat
from mdio.converters.exceptions import GridTraceCountError
from mdio.converters.type_converter import to_structured_type
from mdio.core.config import MDIOSettings
from mdio.core.grid import Grid
from mdio.core.utils_write import MAX_COORDINATES_BYTES
from mdio.core.utils_write import MAX_SIZE_LIVE_MASK
from mdio.core.utils_write import get_constrained_chunksize
from mdio.ingestion.grid_qc import grid_density_qc
from mdio.ingestion.metadata import _add_grid_override_to_metadata
from mdio.ingestion.segy.coordinates import _get_coordinates
from mdio.ingestion.segy.coordinates import _get_spatial_coordinate_unit
from mdio.ingestion.segy.coordinates import _populate_coordinates
from mdio.ingestion.segy.coordinates import _update_template_units
from mdio.ingestion.segy.file_headers import _add_segy_file_headers
from mdio.ingestion.segy.validation import _validate_spec_in_template
from mdio.segy import blocked_io
from mdio.segy.file import get_segy_file_info
from mdio.segy.geometry import GridOverrides
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
    grid_overrides: GridOverrides | None,
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
        template._grid_overrides_applied = True

    # If using NonBinned override, expose non-binned dims as logical coordinates on the template instance
    # and patch _add_coordinates to skip adding them as 1D dimension coordinates
    if grid_overrides is not None and grid_overrides.non_binned and grid_overrides.non_binned_dims:
        non_binned_dims = tuple(grid_overrides.non_binned_dims)
        logger.debug(
            "NonBinned grid override: exposing non-binned dims as coordinates: %s",
            non_binned_dims,
        )
        # Append any missing names; keep existing order and avoid duplicates
        existing = set(template.coordinate_names)
        to_add = tuple(n for n in non_binned_dims if n not in existing)
        if to_add:
            template._logical_coord_names = template._logical_coord_names + to_add
            template._grid_overrides_applied = True

        # Patch _add_coordinates to skip adding non-binned dims as 1D dimension coordinates
        # This prevents them from being added with wrong dimensions (e.g., just "trace")
        # They will be added later by build_dataset with full spatial_dimension_names
        _patch_add_coordinates_for_non_binned(template, set(non_binned_dims))


def _scan_for_headers(
    segy_file_kwargs: SegyFileArguments,
    segy_file_info: SegyFileInfo,
    template: AbstractDatasetTemplate,
    grid_overrides: GridOverrides | None = None,
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


def _coerce_grid_overrides(
    grid_overrides: GridOverrides | dict[str, Any] | None,
) -> GridOverrides | None:
    """Normalize public ``grid_overrides`` input into a :class:`GridOverrides` model.

    The internal ingestion pipeline only accepts the typed model. A legacy ``dict`` is
    converted via :meth:`GridOverrides.from_legacy_dict` and a deprecation message is logged.
    """
    if grid_overrides is None:
        return None

    if isinstance(grid_overrides, GridOverrides):
        return grid_overrides

    logger.warning(
        "Passing `grid_overrides` as a dict is deprecated and will be removed in a "
        "future release; pass a `mdio.GridOverrides` instance instead."
    )
    return GridOverrides.model_validate(grid_overrides)


def segy_to_mdio(  # noqa PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: GridOverrides | dict[str, Any] | None = None,
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
        grid_overrides: Option to add grid overrides. Prefer a :class:`mdio.GridOverrides`
            instance; ``dict`` is still accepted but emits a :class:`DeprecationWarning`.
        segy_header_overrides: Option to override specific SEG-Y headers during ingestion.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
    """
    typed_grid_overrides = _coerce_grid_overrides(grid_overrides)

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
        grid_overrides=typed_grid_overrides,
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

    _add_grid_override_to_metadata(dataset=mdio_ds, grid_overrides=typed_grid_overrides)

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
