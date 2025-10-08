"""Utilities to read, parse, and apply coordinate scalars."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from segy.schema import SegyStandard
from segy.standards.fields import trace as trace_header_fields

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from segy import SegyFile


logger = logging.getLogger(__name__)


COORD_SCALAR_KEY = trace_header_fields.Rev0.COORDINATE_SCALAR.model.name
VALID_COORD_SCALAR = {1, 10, 100, 1000, 10000}
SCALE_COORDINATE_KEYS = [
    "cdp_x",
    "cdp_y",
    "source_coord_x",
    "source_coord_y",
    "group_coord_x",
    "group_coord_y",
]


def _get_coordinate_scalar(segy_file: SegyFile) -> int:
    """Get and parse the coordinate scalar from the first SEG-Y trace header."""
    file_revision = segy_file.spec.segy_standard
    first_header = segy_file.header[0]
    coord_scalar = int(first_header[COORD_SCALAR_KEY])

    # Per Rev2, standardize 0 to 1 if a file is 2+.
    if coord_scalar == 0 and file_revision >= SegyStandard.REV2:
        logger.warning("Coordinate scalar is 0 and file is %s. Setting to 1.", file_revision)
        return 1

    def validate_segy_scalar(scalar: int) -> bool:
        """Validate if coord scalar matches the seg-y standard."""
        logger.debug("Coordinate scalar is %s", scalar)
        return abs(scalar) in VALID_COORD_SCALAR  # valid values

    is_valid = validate_segy_scalar(coord_scalar)
    if not is_valid:
        msg = f"Invalid coordinate scalar: {coord_scalar} for file revision {file_revision}."
        raise ValueError(msg)

    logger.info("Coordinate scalar is parsed as %s", coord_scalar)
    return coord_scalar


def _apply_coordinate_scalar(data: NDArray, scalar: int) -> NDArray:
    if scalar < 0:
        scalar = 1 / scalar
    return data * abs(scalar)
