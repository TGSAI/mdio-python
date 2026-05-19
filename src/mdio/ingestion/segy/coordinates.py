"""Coordinate extraction and unit resolution for SEG-Y ingestion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from segy.standards.codes import MeasurementSystem as SegyMeasurementSystem
from segy.standards.fields import binary as binary_header_fields

from mdio.builder.schemas.v1.units import AngleUnitEnum
from mdio.builder.schemas.v1.units import AngleUnitModel
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.ingestion.coordinates import populate_dim_coordinates
from mdio.ingestion.coordinates import populate_non_dim_coordinates

if TYPE_CHECKING:
    from segy.arrays import HeaderArray as SegyHeaderArray
    from xarray import Dataset as xr_Dataset

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.core.dimension import Dimension
    from mdio.core.grid import Grid
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

    Returns:
        A tuple containing:
            - A list of dimension coordinates (1-D arrays).
            - A dict of non-dimension coordinates (str: N-D arrays).
    """
    dimensions_coords = [grid.select_dim(name) for name in mdio_template.dimension_names]
    non_dim_coords = {name: np.array(segy_headers[name]) for name in mdio_template.coordinate_names}
    return dimensions_coords, non_dim_coords


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
    dataset, drop_vars_delayed = populate_dim_coordinates(dataset, grid, drop_vars_delayed=drop_vars_delayed)

    dataset, drop_vars_delayed = populate_non_dim_coordinates(
        dataset,
        grid,
        coordinates=coords,
        drop_vars_delayed=drop_vars_delayed,
        spatial_coordinate_scalar=spatial_coordinate_scalar,
    )

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
    new_units = {key: AngleUnitModel(angle=AngleUnitEnum.DEGREES) for key in ANGLE_UNIT_KEYS}

    if unit is None:
        template.add_units(new_units)
        return template

    for key in SPATIAL_UNIT_KEYS:
        current_value = template.get_unit_by_key(key)
        if current_value is not None:
            logger.warning("Unit for %s already in template. Will keep the original unit: %s", key, current_value)
            continue

        new_units[key] = unit

    template.add_units(new_units)
    return template
