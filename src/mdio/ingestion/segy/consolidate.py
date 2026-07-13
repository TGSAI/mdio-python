"""Multi-shard SEG-Y -> single MDIO consolidation via region writes.

ADDITIVE feature. This module does NOT change the default single-file
``segy_to_mdio`` behavior. It adds a two-phase workflow for the common case where
one logical dataset (e.g. a shot survey) is split across many SEG-Y files that
should become a single MDIO:

  1. :func:`allocate_mdio_grid` - create the empty global MDIO store from a template
     plus the *global* dimension coordinates (the union across all shards). This
     writes the dataset skeleton (dimension coordinates + fill-valued arrays) once.

  2. :func:`append_segy_shard` - ingest one shard and place its traces into their
     region of the pre-allocated global grid using an in-place (``mode="r+"``) Zarr
     write, leaving all other regions untouched. Call once per shard.

Everything is cloud-native (reads the SEG-Y over the network, writes into the object
store). No SEG-Y bytes are downloaded/concatenated.

Caller (or an LLM agent driving consolidation) MUST guarantee:
  * All shards share the same sample axis, data sample format, revision, and
    header/index byte layout (so one ``segy_spec`` + template applies to all).
  * Every shard's spatial coordinates are a subset of ``global_dimensions``.
  * Shards are DISJOINT and, on the concatenation dimension, occupy a contiguous
    block whose boundaries align to the store's chunk boundaries on that dimension.
    (A boundary chunk shared by two shards would be clobbered by the second write.)

These invariants are exactly what upstream inspection tooling should verify before
consolidation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from segy.config import SegyFileSettings
from zarr import open_group as zarr_open_group

from mdio.api.io import _normalize_path
from mdio.api.io import _normalize_storage_options
from mdio.api.io import to_mdio
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.core.grid import Grid
from mdio.ingestion.dataset_factory import build_mdio_dataset
from mdio.ingestion.schema.resolver import SchemaResolver
from mdio.ingestion.segy.coordinates import get_spatial_coordinate_unit
from mdio.ingestion.segy.coordinates import populate_coordinates
from mdio.ingestion.segy.coordinates import resolve_units
from mdio.ingestion.segy.index_strategies import IndexStrategyRegistry
from mdio.ingestion.segy.raw_headers import build_raw_header_variables
from mdio.ingestion.segy.reader import read_index_headers
from mdio.ingestion.segy.validation import validate_spec_in_template
from mdio.segy import blocked_io
from mdio.segy.file import get_segy_file_info
from mdio.segy.geometry import validate_overrides_for_template
from mdio.segy.utilities import build_mdio_header_type

if TYPE_CHECKING:
    from pathlib import Path

    from segy.config import SegyHeaderOverrides
    from segy.schema import SegySpec
    from upath import UPath

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.core.dimension import Dimension
    from mdio.segy.file import SegyFileArguments
    from mdio.segy.geometry import GridOverrides

logger = logging.getLogger(__name__)


def _resolve_schema(mdio_template: AbstractDatasetTemplate, grid_overrides: GridOverrides | None):
    """Resolve the (format-agnostic) schema for a template + optional grid overrides."""
    schema_effect = IndexStrategyRegistry().schema_effect(grid_overrides)
    return SchemaResolver().resolve(mdio_template, schema_effect)


def allocate_mdio_grid(  # noqa: PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    global_dimensions: list[Dimension],
    output_path: UPath | Path | str,
    reference_segy_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: GridOverrides | None = None,
) -> UPath:
    """Allocate an empty global MDIO store sized to ``global_dimensions``.

    Builds the dataset skeleton (dimension coordinates + fill-valued data/header/mask
    arrays) so shards can later be written into sub-regions with
    :func:`append_segy_shard`.

    Args:
        segy_spec: SEG-Y spec shared by all shards (drives the header dtype).
        mdio_template: MDIO dataset template shared by all shards.
        global_dimensions: Ordered dimensions of the *global* grid, INCLUDING the
            trailing sample/vertical dimension. Coordinates are the union across shards.
        output_path: Output MDIO store path.
        reference_segy_path: Any one shard, used only to derive units/scalar metadata
            for the store (header values are not written here).
        overwrite: Whether to overwrite an existing store.
        grid_overrides: Optional grid override configuration (must match shards).

    Returns:
        The normalized output path of the allocated store.

    Raises:
        FileExistsError: If the store exists and ``overwrite`` is False.
    """
    validate_overrides_for_template(grid_overrides, mdio_template)
    validate_spec_in_template(segy_spec, mdio_template)

    output_path = _normalize_path(output_path)
    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    ref_path = _normalize_path(reference_segy_path)
    ref_kwargs: SegyFileArguments = {
        "url": ref_path.as_posix(),
        "spec": segy_spec,
        "settings": SegyFileSettings(storage_options=ref_path.storage_options),
        "header_overrides": None,
    }
    file_info = get_segy_file_info(ref_kwargs)
    units = resolve_units(mdio_template, get_spatial_coordinate_unit(file_info))

    schema = _resolve_schema(mdio_template, grid_overrides)

    grid = Grid(dims=list(global_dimensions))
    header_dtype = build_mdio_header_type(segy_spec)
    extra_variables = build_raw_header_variables(schema)
    mdio_ds = build_mdio_dataset(
        schema=schema,
        sizes=grid.shape,
        header_dtype=header_dtype,
        units=units,
        extra_variables=extra_variables,
    )

    xr_dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # Write the dimension coordinates (small, global, known up-front). Data, headers and
    # trace_mask stay at their fill values (trace_mask fill == False => all-dead grid).
    for dim in grid.dims:
        xr_dataset[dim.name].values[:] = dim.coords

    to_mdio(xr_dataset, output_path=output_path, mode="w", compute=False)
    dim_names = [dim.name for dim in grid.dims]
    to_mdio(xr_dataset[dim_names], output_path=output_path, mode="r+", compute=True)

    logger.info("Allocated global MDIO grid %s at %s", grid.shape, output_path.as_posix())
    return output_path


def _region_slices_for(dims: tuple[str, ...], region: dict[str, slice]) -> tuple[slice, ...]:
    """Per-variable region slices: use the region slice where the dim applies, else full."""
    return tuple(region.get(name, slice(None)) for name in dims)


def _write_region_vars(
    output_path: UPath,
    xr_dataset,
    region: dict[str, slice],
    var_names: list[str],
) -> None:
    """Write only the shard's sub-region of the given variables via in-place Zarr slicing.

    Mirrors the blocked-I/O pattern (direct Zarr ``array[slices] = values``) so we avoid
    xarray region-write constraints on dimension coordinates.
    """
    storage_options = _normalize_storage_options(output_path)
    zarr_group = zarr_open_group(output_path.as_posix(), mode="r+", storage_options=storage_options)
    for name in var_names:
        da = xr_dataset[name]
        slices = _region_slices_for(tuple(da.dims), region)
        zarr_group[name][slices] = np.asarray(da.values)[slices]


def append_segy_shard(  # noqa: PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    global_dimensions: list[Dimension],
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    grid_overrides: GridOverrides | None = None,
    segy_header_overrides: SegyHeaderOverrides | None = None,
    merge_chunks: set[tuple[int, ...]] | None = None,
) -> dict[str, slice]:
    """Ingest one SEG-Y shard into its region of a pre-allocated global MDIO store.

    The store must already exist (see :func:`allocate_mdio_grid`). The shard's traces are
    placed at their GLOBAL grid positions (the grid map is built by searching the shard's
    header values into ``global_dimensions``), so nothing outside the shard is touched.

    Args:
        segy_spec: SEG-Y spec (same as used for allocation).
        mdio_template: MDIO template (same as used for allocation).
        global_dimensions: The global grid dimensions used at allocation (incl. sample dim).
        input_path: The shard SEG-Y path.
        output_path: The pre-allocated global MDIO store path.
        grid_overrides: Optional grid overrides (same as allocation).
        segy_header_overrides: Optional SEG-Y header overrides for this shard.
        merge_chunks: Optional set of chunk-grid indices this shard shares with other shards;
            these are written read-modify-write to avoid clobbering. Typically
            ``plan.merge_chunks_for(shard_id)`` from :func:`mdio.plan_consolidation`. When
            None, all of this shard's chunks are pure fast writes (safe only if the shard's
            chunks are exclusive to it).

    Returns:
        The spatial region (dim name -> slice) the shard was written into.

    Raises:
        ValueError: If the shard maps to no cells within the global grid.
    """
    validate_overrides_for_template(grid_overrides, mdio_template)
    validate_spec_in_template(segy_spec, mdio_template)

    input_path = _normalize_path(input_path)
    output_path = _normalize_path(output_path)

    segy_file_kwargs: SegyFileArguments = {
        "url": input_path.as_posix(),
        "spec": segy_spec,
        "settings": SegyFileSettings(storage_options=input_path.storage_options),
        "header_overrides": segy_header_overrides,
    }
    file_info = get_segy_file_info(segy_file_kwargs)
    units = resolve_units(mdio_template, get_spatial_coordinate_unit(file_info))
    schema = _resolve_schema(mdio_template, grid_overrides)

    indexed_headers, _shard_dims = read_index_headers(
        segy_file_kwargs=segy_file_kwargs,
        file_info=file_info,
        schema=schema,
        grid_overrides=grid_overrides,
        synthesize_dims=mdio_template.synthesize_missing_dims,
        template=mdio_template,
    )

    # Build the grid on the GLOBAL dimensions; build_map searchsorts the shard's header
    # values into the global coords, so live cells land at their global positions.
    grid = Grid(dims=list(global_dimensions))
    grid.build_map(indexed_headers)

    live_mask = np.asarray(grid.live_mask[:])
    nonzero = np.argwhere(live_mask)
    if nonzero.size == 0:
        err = "Shard maps to no cells within the global grid; check global_dimensions and index map."
        raise ValueError(err)

    spatial_dim_names = grid.dim_names[:-1]
    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0) + 1
    region = {name: slice(int(mins[i]), int(maxs[i])) for i, name in enumerate(spatial_dim_names)}
    logger.info("Appending shard %s into region %s", input_path.as_posix(), region)

    # Build a global in-memory xarray dataset (for encoding + coordinate population); do NOT
    # re-create the store.
    header_dtype = build_mdio_header_type(segy_spec)
    extra_variables = build_raw_header_variables(schema)
    mdio_ds = build_mdio_dataset(
        schema=schema,
        sizes=grid.shape,
        header_dtype=header_dtype,
        units=units,
        extra_variables=extra_variables,
    )
    xr_dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    non_dim_coords = {}
    for coord in schema.coordinates:
        if coord.name in indexed_headers.dtype.names:
            non_dim_coords[coord.name] = np.array(indexed_headers[coord.name])

    xr_dataset, drop_vars_delayed = populate_coordinates(
        dataset=xr_dataset,
        grid=grid,
        coords=non_dim_coords,
        spatial_coordinate_scalar=file_info.coordinate_scalar,
    )
    xr_dataset.trace_mask.data[:] = grid.live_mask

    # Region-scoped write of trace_mask + non-dim coordinates (dimension coordinates were
    # written once at allocation and are excluded here).
    non_dim_coord_names = [c.name for c in schema.coordinates if c.name in xr_dataset]
    _write_region_vars(output_path, xr_dataset, region, var_names=non_dim_coord_names + ["trace_mask"])

    # Data + structured headers: blocked-I/O writes only the shard's live chunks (mode="r+")
    # using the global grid map, so other shards' regions are untouched.
    xr_dataset = xr_dataset.drop_vars(drop_vars_delayed)
    blocked_io.to_zarr(
        segy_file_kwargs=segy_file_kwargs,
        output_path=output_path,
        grid_map=grid.map,
        dataset=xr_dataset,
        data_variable_name=schema.default_variable_name,
        merge_chunks=merge_chunks,
    )

    return region
