"""Generic coordinate population for ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mdio.segy.scalar import SCALE_COORDINATE_KEYS
from mdio.segy.scalar import _apply_coordinate_scalar

if TYPE_CHECKING:
    from segy.arrays import HeaderArray as SegyHeaderArray
    from xarray import Dataset as xr_Dataset

    from mdio.core.grid import Grid


def populate_dim_coordinates(
    dataset: xr_Dataset, grid: Grid, drop_vars_delayed: list[str]
) -> tuple[xr_Dataset, list[str]]:
    """Populate the xarray dataset with dimension coordinate variables."""
    for dim in grid.dims:
        dataset[dim.name].values[:] = dim.coords
        drop_vars_delayed.append(dim.name)
    return dataset, drop_vars_delayed


def _extract_coord_trace_indices(grid: Grid, coord_axes: tuple[int, ...]) -> np.ndarray:
    """Return the minimum live trace index per coordinate cell across any reduced spatial axes.

    Unpopulated cells in ``grid.map`` hold ``grid.map.fill_value`` (``iinfo(dtype).max``), so
    taking the minimum across reduced axes selects a valid live trace index whenever any trace
    is live along those axes, and preserves ``fill_value`` only when all traces are dead.
    """
    reduced_axes = tuple(idx for idx in range(grid.map.ndim) if idx not in coord_axes)
    if not reduced_axes:
        return np.asarray(grid.map[:])

    coord_shape = tuple(grid.map.shape[axis] for axis in coord_axes)
    coord_trace_indices = np.full(coord_shape, grid.map.fill_value, dtype=grid.map.dtype)

    for block_id in np.ndindex(*grid.map.cdata_shape):
        target_slice = tuple(
            slice(
                block_id[axis] * grid.map.chunks[axis],
                min((block_id[axis] + 1) * grid.map.chunks[axis], grid.map.shape[axis]),
            )
            for axis in coord_axes
        )
        block_min = np.asarray(grid.map.blocks[block_id]).min(axis=reduced_axes)
        coord_trace_indices[target_slice] = np.minimum(coord_trace_indices[target_slice], block_min)

    return coord_trace_indices


def populate_non_dim_coordinates(
    dataset: xr_Dataset,
    grid: Grid,
    coordinates: dict[str, SegyHeaderArray],
    drop_vars_delayed: list[str],
    spatial_coordinate_scalar: int,
) -> tuple[xr_Dataset, list[str]]:
    """Populate the xarray dataset with coordinate variables.

    Memory optimization: Processes coordinates one at a time and caches reduced trace index maps
    per coordinate-axis tuple.
    """
    non_data_domain_dims = grid.dim_names[:-1]
    trace_indices_by_axes: dict[tuple[int, ...], np.ndarray] = {}

    coord_names = list(coordinates.keys())
    for coord_name in coord_names:
        coord_values = coordinates.pop(coord_name)
        da_coord = dataset[coord_name]

        coord_shape = da_coord.shape

        fill_value = da_coord.encoding.get("_FillValue") or da_coord.encoding.get("fill_value")
        if fill_value is None:
            fill_value = np.nan
        tmp_coord_values = np.full(coord_shape, fill_value, dtype=da_coord.dtype)

        coord_axes = tuple(non_data_domain_dims.index(coord_dim) for coord_dim in da_coord.dims)
        if coord_axes not in trace_indices_by_axes:
            trace_indices_by_axes[coord_axes] = _extract_coord_trace_indices(grid, coord_axes)
        coord_trace_indices = trace_indices_by_axes[coord_axes]

        not_null = coord_trace_indices != grid.map.fill_value

        if not_null.any():
            valid_indices = coord_trace_indices[not_null]
            tmp_coord_values[not_null] = coord_values[valid_indices]

        if coord_name in SCALE_COORDINATE_KEYS:
            tmp_coord_values = _apply_coordinate_scalar(tmp_coord_values, spatial_coordinate_scalar)

        dataset[coord_name][:] = tmp_coord_values
        drop_vars_delayed.append(coord_name)

        del tmp_coord_values, not_null, coord_values

        # TODO(Altay): Add verification of reduced coordinates being consistent across live traces
        # https://github.com/TGSAI/mdio-python/issues/645

    return dataset, drop_vars_delayed
