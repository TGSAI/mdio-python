"""Serializer that formats the populated xarray dataset and writes it to MDIO / Zarr."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from mdio.api.io import to_mdio
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.ingestion.segy.coordinates import populate_coordinates
from mdio.ingestion.segy.file_headers import add_segy_file_headers
from mdio.segy import blocked_io

if TYPE_CHECKING:
    from upath import UPath

    from mdio.builder.schemas import Dataset
    from mdio.core.grid import Grid
    from mdio.ingestion.schema import ResolvedSchema
    from mdio.segy.file import SegyFileArguments
    from mdio.segy.file import SegyFileInfo

logger = logging.getLogger(__name__)


def serialize_to_mdio(  # noqa: PLR0913
    mdio_ds: Dataset,
    grid: Grid,
    indexed_headers: np.ndarray,
    schema: ResolvedSchema,
    file_info: SegyFileInfo,
    segy_file_kwargs: SegyFileArguments,
    output_path: UPath,
) -> None:
    """Serialize the constructed MDIO Dataset and populate it with trace data.

    This function coordinates coordinate population, trace mask writing, file headers,
    staged to_mdio Zarr writes, and the final blocked-I/O writing.

    Args:
        mdio_ds: The constructed Dataset schema model.
        grid: The built ingestion Grid.
        indexed_headers: Transformed trace headers.
        schema: Resolved schema of the dataset.
        file_info: Metadata info of the SEG-Y file.
        segy_file_kwargs: SEG-Y file arguments.
        output_path: Universal output path for writing.
    """
    # 1. Extract non-dimension coordinates from transformed headers
    non_dim_coords = {}
    for coord in schema.coordinates:
        if coord.name in indexed_headers.dtype.names:
            non_dim_coords[coord.name] = np.array(indexed_headers[coord.name])
        else:
            logger.warning(
                "Coordinate '%s' expected by schema but not found in transformed headers.",
                coord.name,
            )

    # 2. Convert to xarray dataset
    xr_dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # 3. Populate coordinates with grid mappings
    xr_dataset, drop_vars_delayed = populate_coordinates(
        dataset=xr_dataset,
        grid=grid,
        coords=non_dim_coords,
        spatial_coordinate_scalar=file_info.coordinate_scalar,
    )

    # 4. Attach SEG-Y text and binary file headers
    xr_dataset = add_segy_file_headers(xr_dataset, file_info)

    # 5. Populate trace mask
    xr_dataset.trace_mask.data[:] = grid.live_mask

    # 6. Initialize Zarr store with metadata (staged empty arrays)
    to_mdio(xr_dataset, output_path=output_path, mode="w", compute=False)

    # 7. Write delayed coordinate variables and trace mask
    unindexed_dims = [d for d in xr_dataset.dims if d not in xr_dataset.coords]
    for d in unindexed_dims:
        if d in drop_vars_delayed:
            drop_vars_delayed.remove(d)

    meta_ds = xr_dataset[drop_vars_delayed + ["trace_mask"]]
    to_mdio(meta_ds, output_path=output_path, mode="r+", compute=True)

    # Drop delayed vars to prepare for blocked-I/O writing
    xr_dataset = xr_dataset.drop_vars(drop_vars_delayed)

    # 8. Write bulk trace data and structured headers via blocked-I/O workers
    blocked_io.to_zarr(
        segy_file_kwargs=segy_file_kwargs,
        output_path=output_path,
        grid_map=grid.map,
        dataset=xr_dataset,
        data_variable_name=schema.default_variable_name,
    )
