"""Utilities to read, parse, and apply coordinate scalars."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from segy.standards.fields import trace as trace_header_fields

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


def _apply_coordinate_scalar(data: NDArray, scalar: int) -> NDArray:
    if scalar < 0:
        scalar = 1 / scalar
    return data * abs(scalar)
