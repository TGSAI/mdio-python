"""This module defines `LabeledArray`, `Coordinate`, and `Variable`.

`LabeledArray` is a basic array unit which includes basic properties like
name, dimension, data type, compressor etc.

`Coordinate` extends the `LabeledArray` class, it represents the Coordinate
array in the MDIO format. It has dimensions which are fully defined and can hold
additional metadata.

`Variable` is another class that extends the `LabeledArray`. It represents a
variable in MDIO format. It can have coordinates and can also hold metadata.
"""

from typing import Any

from pydantic import Field

from mdio.builder.schemas.base import NamedArray
from mdio.builder.schemas.chunk_grid import RectilinearChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.core import CamelCaseStrictModel
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.stats import SummaryStatistics
from mdio.builder.schemas.v1.units import AllUnitModel


class CoordinateMetadata(CamelCaseStrictModel):
    """Reduced Metadata, perfect for simple Coordinates."""

    units_v1: AllUnitModel | None = Field(default=None)
    attributes: dict[str, Any] | None = Field(default=None)


class VariableMetadata(CoordinateMetadata):
    """Complete Metadata for Variables and complex or large Coordinates."""

    chunk_grid: RegularChunkGrid | RectilinearChunkGrid | None = Field(
        default=None,
        description="Chunk grid specification for the array.",
    )

    stats_v1: SummaryStatistics | list[SummaryStatistics] | None = Field(
        default=None,
        description="Minimal summary statistics.",
    )


class Coordinate(NamedArray):
    """A simple MDIO Coordinate array with metadata.

    For large or complex Coordinates, define a Variable instead.
    """

    data_type: ScalarType = Field(..., description="Data type of Coordinate.")
    metadata: CoordinateMetadata | None = Field(default=None, description="Coordinate Metadata.")


class Variable(NamedArray):
    """An MDIO Variable that has coordinates and metadata."""

    coordinates: list[Coordinate] | list[str] | None = Field(
        default=None,
        description="Coordinates of the MDIO Variable dimensions.",
    )
    metadata: VariableMetadata | None = Field(default=None, description="Variable Metadata.")
