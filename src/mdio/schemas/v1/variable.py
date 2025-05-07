"""This module defines `LabeledArray`, `Coordinate`, and `Variable`.

`LabeledArray` is a basic array unit which includes basic properties like
name, dimension, data type, compressor etc.

`Coordinate` extends the `LabeledArray` class, it represents the Coordinate
array in the MDIO format. It has dimensions which are fully defined and can hold
additional metadata.

`Variable` is another class that extends the `LabeledArray`. It represents a
variable in MDIO format. It can have coordinates and can also hold metadata.
"""

from pydantic import Field
from pydantic import create_model

from mdio.schemas.base import NamedArray
from mdio.schemas.core import CamelCaseStrictModel
from mdio.schemas.core import model_fields
from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.units import AllUnits


class Coordinate(NamedArray):
    """An MDIO coordinate array with metadata."""

    data_type: ScalarType = Field(..., description="Data type of coordinate.")
    metadata: list[AllUnits | UserAttributes] | None = Field(
        default=None, description="Coordinate metadata."
    )


VariableMetadata = create_model(
    "VariableMetadata",
    **model_fields(ChunkGridMetadata),
    **model_fields(AllUnits),
    **model_fields(StatisticsMetadata),
    **model_fields(UserAttributes),
    __base__=CamelCaseStrictModel,
)


class Variable(NamedArray):
    """An MDIO variable that has coordinates and metadata."""

    coordinates: list[Coordinate] | list[str] | None = Field(
        default=None,
        description="Coordinates of the MDIO variable dimensions.",
    )
    metadata: VariableMetadata | None = Field(default=None, description="Variable metadata.")
