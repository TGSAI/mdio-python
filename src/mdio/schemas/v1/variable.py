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

CoordinateMetadata = create_model(
    "CoordinateMetadata",
    **model_fields(AllUnits),
    **model_fields(UserAttributes),
    __base__=CamelCaseStrictModel,
    __doc__="Reduced Metadata, perfect for simple Coordinates.",
)


class Coordinate(NamedArray):
    """A simple MDIO Coordinate array with metadata.

    For large or complex Coordinates, define a Variable instead.
    """

    data_type: ScalarType = Field(..., description="Data type of Coordinate.")
    metadata: CoordinateMetadata | None = Field(default=None, description="Coordinate Metadata.")


VariableMetadata = create_model(
    "VariableMetadata",
    **model_fields(ChunkGridMetadata),
    **model_fields(AllUnits),
    **model_fields(StatisticsMetadata),
    **model_fields(UserAttributes),
    __base__=CamelCaseStrictModel,
    __doc__="Complete Metadata for Variables and complex or large Coordinates.",
)


class Variable(NamedArray):
    """An MDIO Variable that has coordinates and metadata."""

    coordinates: list[Coordinate] | list[str] | None = Field(
        default=None,
        description="Coordinates of the MDIO Variable dimensions.",
    )
    metadata: VariableMetadata | None = Field(default=None, description="Variable Metadata.")
