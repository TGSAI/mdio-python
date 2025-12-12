"""Ingestion Pipeline for SEG-Y to MDIO.

This module implements the ingestion pipeline with clear separation of concerns:
1. Schema Resolution Phase
2. Header Analysis Phase
3. Index Strategy Phase
4. Grid Building Phase
5. Dataset Building Phase
6. Data Writing Phase
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import zarr

from mdio.api.io import _normalize_path
from mdio.api.io import to_mdio
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.constants import ZarrFormat
from mdio.converters.exceptions import GridTraceCountError
from mdio.converters.type_converter import to_structured_type
from mdio.core.config import MDIOSettings
from mdio.core.grid import Grid
from mdio.ingestion.coordinate_utils import get_coordinates
from mdio.ingestion.coordinate_utils import get_spatial_coordinate_unit
from mdio.ingestion.coordinate_utils import populate_dim_coordinates
from mdio.ingestion.coordinate_utils import populate_non_dim_coordinates
from mdio.ingestion.coordinate_utils import update_template_units
from mdio.ingestion.dataset_factory import DatasetFactory
from mdio.ingestion.header_analyzer import HeaderAnalyzer
from mdio.ingestion.index_strategies import IndexStrategyFactory
from mdio.ingestion.metadata import add_grid_override_to_metadata
from mdio.ingestion.metadata import add_segy_file_headers
from mdio.ingestion.schema_resolver import SchemaResolver
from mdio.ingestion.validation import grid_density_qc
from mdio.ingestion.validation import validate_spec_in_template
from mdio.segy import blocked_io
from mdio.segy.file import get_segy_file_info

if TYPE_CHECKING:
    from pathlib import Path

    from segy.config import SegyHeaderOverrides
    from segy.schema import SegySpec
    from upath import UPath

    from mdio.builder.schemas.v1.dataset import Dataset
    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.segy.file import SegyFileArguments
    from mdio.segy.geometry import GridOverrides

logger = logging.getLogger(__name__)


def run_segy_ingestion(  # noqa PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: GridOverrides | None = None,
    segy_header_overrides: SegyHeaderOverrides | None = None,
) -> None:
    """Convert SEG-Y to MDIO using the refactored pipeline.

    This ingestion pipeline has clear separation of concerns.
    It follows an explicit 6-phase approach:

    1. Schema Resolution: Resolve template + overrides → final schema
    2. Header Analysis: Extract only required headers
    3. Index Strategy: Transform headers using appropriate strategy
    4. Grid Building: Build and validate grid from dimensions
    5. Dataset Building: Build dataset from schema
    6. Data Writing: Write data to Zarr store

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
        ValueError: If required fields are missing from segy_spec.
    """
    settings = MDIOSettings()

    logger.info("Running ingestion pipeline")

    # Validate spec has required fields
    validate_spec_in_template(segy_spec, mdio_template)

    # Normalize paths
    input_path = _normalize_path(input_path)
    output_path = _normalize_path(output_path)

    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    # Prepare SEG-Y file arguments
    from segy.config import SegyFileSettings

    segy_settings = SegyFileSettings(storage_options=input_path.storage_options)
    segy_file_kwargs: SegyFileArguments = {
        "url": input_path.as_posix(),
        "spec": segy_spec,
        "settings": segy_settings,
        "header_overrides": segy_header_overrides,
    }

    # Get SEG-Y file info
    segy_file_info = get_segy_file_info(segy_file_kwargs)

    # ============================================================
    # PHASE 1: Schema Resolution
    # ============================================================
    logger.info("Phase 1: Resolving schema from template and overrides")
    resolver = SchemaResolver()
    schema = resolver.resolve(mdio_template, grid_overrides)
    logger.info("Resolved schema: %s dimensions, %s coordinates", len(schema.dimensions), len(schema.coordinates))

    # ============================================================
    # PHASE 2: Header Analysis
    # ============================================================
    logger.info("Phase 2: Analyzing headers")
    analyzer = HeaderAnalyzer()
    requirements = analyzer.requirements_from_schema(schema)
    logger.info("Required header fields: %s", sorted(requirements.required_fields))

    raw_headers = analyzer.analyze(
        segy_file_kwargs=segy_file_kwargs,
        requirements=requirements,
        num_traces=segy_file_info.num_traces,
    )

    # ============================================================
    # PHASE 3: Index Strategy
    # ============================================================
    logger.info("Phase 3: Applying index strategy")
    strategy_factory = IndexStrategyFactory()
    strategy = strategy_factory.create_strategy(grid_overrides)
    logger.info("Using strategy: %s", strategy.name)

    indexed_headers = strategy.transform_headers(raw_headers)
    dim_names = tuple(d.name for d in schema.dimensions if d.is_spatial)
    dimensions = strategy.compute_dimensions(indexed_headers, dim_names)

    # Add vertical dimension from file info
    from mdio.core import Dimension

    sample_labels = segy_file_info.sample_labels / 1000  # normalize
    if all(sample_labels.astype("int64") == sample_labels):
        sample_labels = sample_labels.astype("int64")

    vertical_dim_name = schema.dimensions[-1].name
    vertical_dim = Dimension(coords=sample_labels, name=vertical_dim_name)
    dimensions.append(vertical_dim)

    logger.info("Computed %d dimensions", len(dimensions))

    # ============================================================
    # PHASE 4: Grid Building and Validation
    # ============================================================
    logger.info("Phase 4: Building and validating grid")
    grid = Grid(dims=dimensions)
    grid_density_qc(grid, segy_file_info.num_traces)
    grid.build_map(indexed_headers)

    # Validate grid
    live_trace_count = int(np.sum(grid.live_mask))
    if live_trace_count != segy_file_info.num_traces:
        for dim_name in grid.dim_names:
            dim_min, dim_max = grid.get_min(dim_name), grid.get_max(dim_name)
            logger.warning("%s min: %s max: %s", dim_name, dim_min, dim_max)
        logger.warning("Ingestion grid shape: %s.", grid.shape)
        raise GridTraceCountError(live_trace_count, segy_file_info.num_traces)

    logger.info("Grid validated: shape=%s, live_traces=%d", grid.shape, live_trace_count)

    # ============================================================
    # PHASE 5: Dataset Building
    # ============================================================
    logger.info("Phase 5: Building dataset")

    # Get coordinates for dataset
    _, non_dim_coords = get_coordinates(grid, indexed_headers, mdio_template)

    # Get header dtype
    header_dtype = to_structured_type(segy_spec.trace.header.dtype)

    # Determine if raw headers should be included
    include_raw_headers = False
    if settings.raw_headers:
        if zarr.config.get("default_zarr_format") == ZarrFormat.V2:
            logger.warning("Raw headers are only supported for Zarr v3. Skipping raw headers.")
        else:
            logger.warning("MDIO__IMPORT__RAW_HEADERS is experimental and expected to change or be removed.")
            include_raw_headers = True

    # Build dataset using factory
    factory = DatasetFactory()
    mdio_ds: Dataset = factory.build(
        schema=schema,
        dimensions=dimensions,
        header_dtype=header_dtype,
        include_raw_headers=include_raw_headers,
    )

    # Add grid override metadata
    add_grid_override_to_metadata(dataset=mdio_ds, grid_overrides=grid_overrides)

    # Update template units (for spatial coordinates)
    spatial_unit = get_spatial_coordinate_unit(segy_file_info)
    mdio_template = update_template_units(mdio_template, spatial_unit)

    logger.info("Dataset built successfully")

    # ============================================================
    # PHASE 6: Data Writing
    # ============================================================
    logger.info("Phase 6: Writing data to Zarr store")

    # Convert to xarray
    xr_dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # Populate coordinates
    drop_vars_delayed = []
    xr_dataset, drop_vars_delayed = populate_dim_coordinates(xr_dataset, grid, drop_vars_delayed)
    xr_dataset, drop_vars_delayed = populate_non_dim_coordinates(
        xr_dataset,
        grid,
        non_dim_coords,
        drop_vars_delayed,
        segy_file_info.coordinate_scalar,
    )

    # Add SEG-Y file headers if requested
    if settings.save_segy_file_header:
        xr_dataset = add_segy_file_headers(xr_dataset, segy_file_info)

    # Set trace mask
    xr_dataset.trace_mask.data[:] = grid.live_mask

    # Create Zarr store with empty arrays
    to_mdio(xr_dataset, output_path=output_path, mode="w", compute=False)

    # Write non-dimension coordinates and trace mask
    unindexed_dims = [d for d in xr_dataset.dims if d not in xr_dataset.coords]
    for d in unindexed_dims:
        if d in drop_vars_delayed:
            drop_vars_delayed.remove(d)

    meta_ds = xr_dataset[drop_vars_delayed + ["trace_mask"]]
    to_mdio(meta_ds, output_path=output_path, mode="r+", compute=True)

    # Drop written variables
    xr_dataset = xr_dataset.drop_vars(drop_vars_delayed)

    # Write headers and traces in chunks
    default_variable_name = schema.default_variable_name
    blocked_io.to_zarr(
        segy_file_kwargs=segy_file_kwargs,
        output_path=output_path,
        grid_map=grid.map,
        dataset=xr_dataset,
        data_variable_name=default_variable_name,
    )

    logger.info("Ingestion complete!")
