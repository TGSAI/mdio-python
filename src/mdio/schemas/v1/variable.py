"""This module defines `LabeledArray`, `Coordinate`, and `Variable`.

`LabeledArray` is a basic array unit which includes basic properties like
name, dimension, data type, compressor etc.

`Coordinate` extends the `LabeledArray` class, it represents the Coordinate
array in the MDIO format. It has dimensions which are fully defined and can hold
additional metadata.

`Variable` is another class that extends the `LabeledArray`. It represents a
variable in MDIO format. It can have coordinates and can also hold metadata.
"""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from mdio.schemas.base import Compressors
from mdio.schemas.base import DataType
from mdio.schemas.base import Dimension
from mdio.schemas.base import StructuredDataType
from mdio.schemas.base import UserAttributes
from mdio.schemas.v1 import CoordinateUnits
from mdio.schemas.v1 import Units
from mdio.schemas.v1.stats import SummaryStatisticsMetadata


class LabeledArray(BaseModel):
    """An array with more metadata and labels."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the array.")
    long_name: str = Field(default=None, description="Fully descriptive name.")
    dimensions: list[Dimension] = Field(
        ..., description="List of dimensions and respective sizes."
    )
    dtype: DataType | StructuredDataType = Field(
        ..., description="Numeric or structured type of variable."
    )
    compressor: Compressors | None = Field(
        default=None, description="Compression settings."
    )


class Coordinate(LabeledArray):
    """An MDIO coordinate array with metadata."""

    dimensions: list[str] = Field(
        ...,
        description="List of dimension names that maps to fully defined dimensions.",
    )
    dtype: DataType = Field(..., description="Data type of coordinate.")
    metadata: list[CoordinateUnits | UserAttributes] | None = Field(
        default=None, description="Coordinate metadata."
    )


class Variable(LabeledArray):
    """An MDIO variable that has coordinates and metadata."""

    coordinates: list[Coordinate] | None = Field(
        default=None,
        description="Coordinates (labels) of the MDIO variable dimensions.",
    )
    metadata: list[Units | SummaryStatisticsMetadata | UserAttributes] | None = Field(
        default=None, description="Variable metadata."
    )
