"""Ingestion pipeline for SEG-Y to MDIO."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from segy.config import SegyFileSettings

from mdio.api.io import _normalize_path
from mdio.converters.exceptions import GridTraceCountError
from mdio.converters.type_converter import to_structured_type
from mdio.core.grid import Grid
from mdio.ingestion.dataset_factory import build_mdio_dataset
from mdio.ingestion.grid_qc import grid_density_qc
from mdio.ingestion.metadata import add_grid_override_to_metadata
from mdio.ingestion.schema.resolver import SchemaResolver
from mdio.ingestion.segy.coordinates import get_spatial_coordinate_unit
from mdio.ingestion.segy.coordinates import resolve_units
from mdio.ingestion.segy.raw_headers import build_raw_header_variables
from mdio.ingestion.segy.reader import read_index_headers
from mdio.ingestion.segy.serializer import serialize_to_mdio
from mdio.ingestion.segy.validation import validate_spec_in_template
from mdio.segy.file import get_segy_file_info
from mdio.segy.geometry import validate_overrides_for_template

if TYPE_CHECKING:
    from pathlib import Path

    from segy.config import SegyHeaderOverrides
    from segy.schema import SegySpec
    from upath import UPath

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.core.dimension import Dimension
    from mdio.ingestion.schema import ResolvedSchema
    from mdio.segy.file import SegyFileArguments
    from mdio.segy.geometry import GridOverrides

logger = logging.getLogger(__name__)


def _resolve_output_path(output_path: UPath | Path | str, overwrite: bool) -> UPath:
    """Normalize output path and verify overwrite policy.

    Args:
        output_path: Output location path.
        overwrite: Whether to allow overwriting an existing directory.

    Returns:
        Normalized output path.

    Raises:
        FileExistsError: If output path exists and overwrite is False.
    """
    output_path = _normalize_path(output_path)
    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)
    return output_path


def _verify_calculated_dimensions(
    schema: ResolvedSchema,
    dimensions: list[Dimension],
    template_name: str,
) -> None:
    """Verify all calculated dimensions required by the schema were produced.

    Args:
        schema: Resolved dataset schema.
        dimensions: Dimensions produced during header analysis.
        template_name: Name of the dataset template.

    Raises:
        ValueError: If a required calculated dimension is missing.
    """
    missing = schema.missing_calculated_dimensions(dim.name for dim in dimensions)
    if missing:
        err = (
            f"Required computed fields {sorted(missing)} for template {template_name} not found "
            f"after grid overrides. Please ensure the correct grid overrides are applied."
        )
        raise ValueError(err)


def _build_grid(dimensions: list[Dimension], indexed_headers: np.ndarray, num_traces: int) -> Grid:
    """Build the ingestion grid, run density QC, and verify live trace count.

    Args:
        dimensions: Ordered dimensions.
        indexed_headers: Transformed trace headers.
        num_traces: Expected trace count.

    Returns:
        Built and validated ingestion grid.

    Raises:
        GridTraceCountError: If live trace count does not match expected trace count.
    """
    grid = Grid(dims=dimensions)
    grid_density_qc(grid, num_traces)
    grid.build_map(indexed_headers)

    live_trace_count = int(np.sum(grid.live_mask))
    if live_trace_count != num_traces:
        for dim_name in grid.dim_names:
            logger.warning("%s min: %s max: %s", dim_name, grid.get_min(dim_name), grid.get_max(dim_name))
        logger.warning("Ingestion grid shape: %s.", grid.shape)
        raise GridTraceCountError(live_trace_count, num_traces)

    return grid


def segy_to_mdio(  # noqa: PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: GridOverrides | None = None,
    segy_header_overrides: SegyHeaderOverrides | None = None,
) -> None:
    """Convert SEG-Y file to MDIO dataset.

    Args:
        segy_spec: SEG-Y file specification.
        mdio_template: MDIO dataset template.
        input_path: Input SEG-Y file path.
        output_path: Output MDIO dataset path.
        overwrite: Whether to overwrite existing output.
        grid_overrides: Grid override configuration.
        segy_header_overrides: Specific header overrides.

    Raises:  # noqa: DOC502
        FileExistsError: If output path exists and overwrite is False.
        ValueError: If required fields are missing or required calculated dimensions are missing.
        GridTraceCountError: If built grid live trace count does not match SEG-Y file trace count.
    """
    validate_overrides_for_template(grid_overrides, mdio_template)
    validate_spec_in_template(segy_spec, mdio_template)

    input_path = _normalize_path(input_path)
    output_path = _resolve_output_path(output_path, overwrite)

    segy_file_kwargs: SegyFileArguments = {
        "url": input_path.as_posix(),
        "spec": segy_spec,
        "settings": SegyFileSettings(storage_options=input_path.storage_options),
        "header_overrides": segy_header_overrides,
    }
    segy_file_info = get_segy_file_info(segy_file_kwargs)

    spatial_unit = get_spatial_coordinate_unit(segy_file_info)
    units = resolve_units(mdio_template, spatial_unit)

    schema = SchemaResolver().resolve(mdio_template, grid_overrides)

    indexed_headers, dimensions = read_index_headers(
        segy_file_kwargs=segy_file_kwargs,
        file_info=segy_file_info,
        schema=schema,
        grid_overrides=grid_overrides,
        synthesize_dims=mdio_template.synthesize_missing_dims,
        template=mdio_template,
    )
    _verify_calculated_dimensions(schema, dimensions, mdio_template.name)

    grid = _build_grid(dimensions, indexed_headers, segy_file_info.num_traces)

    header_dtype = to_structured_type(segy_spec.trace.header.dtype)
    extra_variables = build_raw_header_variables(schema)
    mdio_ds = build_mdio_dataset(
        schema=schema,
        sizes=grid.shape,
        header_dtype=header_dtype,
        units=units,
        extra_variables=extra_variables,
    )
    add_grid_override_to_metadata(dataset=mdio_ds, grid_overrides=grid_overrides)

    serialize_to_mdio(
        mdio_ds=mdio_ds,
        grid=grid,
        indexed_headers=indexed_headers,
        schema=schema,
        file_info=segy_file_info,
        segy_file_kwargs=segy_file_kwargs,
        output_path=output_path,
    )
