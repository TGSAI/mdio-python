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
    non_data_domain_dims = grid.dim_names[:-1]

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
        coord_slices = tuple(slice(None) if idx in coord_axes else 0 for idx in range(len(non_data_domain_dims)))

        coord_trace_indices = np.asarray(grid.map[coord_slices])

        not_null = coord_trace_indices != grid.map.fill_value

        if not_null.any():
            valid_indices = coord_trace_indices[not_null]
            tmp_coord_values[not_null] = coord_values[valid_indices]

        if coord_name in SCALE_COORDINATE_KEYS:
            tmp_coord_values = _apply_coordinate_scalar(tmp_coord_values, spatial_coordinate_scalar)

        dataset[coord_name][:] = tmp_coord_values
        drop_vars_delayed.append(coord_name)

        del tmp_coord_values, coord_trace_indices, not_null, coord_values

        # TODO(Altay): Add verification of reduced coordinates being the same as the first
        # https://github.com/TGSAI/mdio-python/issues/645

    return dataset, drop_vars_delayed
