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
    """Normalize the output path and enforce the overwrite policy.

    Args:
        output_path: Requested output location.
        overwrite: Whether an existing location may be overwritten.

    Returns:
        The normalized output path.

    Raises:
        FileExistsError: If the location exists and ``overwrite`` is False.
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
    """Ensure every calculated dimension required by the schema was produced.

    Args:
        schema: The resolved schema.
        dimensions: Dimensions produced by header analysis.
        template_name: Template name, used only for the error message.

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
    """Build the ingestion grid, run density QC, and verify the live trace count.

    Args:
        dimensions: Ordered spatial + vertical dimensions.
        indexed_headers: Transformed trace headers used to build the grid map.
        num_traces: Number of traces reported by the SEG-Y file.

    Returns:
        The built and validated grid.

    Raises:
        GridTraceCountError: If the live trace count does not match ``num_traces``.
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


def run_segy_ingestion(  # noqa: PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: GridOverrides | None = None,
    segy_header_overrides: SegyHeaderOverrides | None = None,
) -> None:
    """Convert SEG-Y to MDIO.

    Pipeline phases: schema resolution, header analysis, index strategy, grid build,
    dataset build, data write.

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        mdio_template: The MDIO template to use for the conversion.
        input_path: The universal path of the input SEG-Y file.
        output_path: The universal path for the output MDIO v1 file.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.
        grid_overrides: Grid override configuration for non-standard geometries.
        segy_header_overrides: Option to override specific SEG-Y headers during ingestion.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
        ValueError: If required fields are missing from segy_spec or required computed
            dimensions are not produced after grid overrides are applied.
        GridTraceCountError: If the live trace count in the built grid does not match
            the number of traces reported by the SEG-Y file.
    """
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
