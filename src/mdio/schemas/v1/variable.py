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
from typing import get_type_hints

from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

from mdio.schemas.base.array import NamedArray
from mdio.schemas.base.dtype import ScalarType
from mdio.schemas.base.dtype import StructuredType
from mdio.schemas.base.encoding import ChunkGridMetadata
from mdio.schemas.base.metadata import MetadataContainer
from mdio.schemas.base.metadata import UserAttributes
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import CoordinateUnits


class Coordinate(NamedArray):
    """An MDIO coordinate array with metadata."""

    data_type: ScalarType = Field(..., description="Data type of coordinate.")
    metadata: list[CoordinateUnits | UserAttributes] | None = Field(
        default=None, description="Coordinate metadata."
    )


# Function to extract field types and defaults
def model_fields(model: type[BaseModel]) -> dict[str, tuple[Any, Any]]:
    """Extract Pydantic BaseModel fields.

    Args:
        model: (Type) The model object for which the fields will be extracted.

    Returns:
        A dictionary containing the fields of the model along with
        their corresponding types and default values.

    Example:
        >>> class MyModel(BaseModel):
        ...     name: str
        ...     age: int = 0
        ...
        >>> model_fields(MyModel)
        {'name': (str, <default_value>), 'age': (int, 0)}
    """
    annotations = get_type_hints(model)

    fields = {}
    for field_name, field in model.model_fields.items():
        fields[field_name] = (annotations[field_name], field)

    return fields


VariableMetadata = create_model(
    "VariableMetadata",
    **model_fields(ChunkGridMetadata),
    **model_fields(AllUnits),
    **model_fields(StatisticsMetadata),
    **model_fields(UserAttributes),
    __base__=MetadataContainer,
)


class Variable(NamedArray):
    """An MDIO variable that has coordinates and metadata."""

    data_type: ScalarType | StructuredType = Field(
        ..., description="Type of the array."
    )

    coordinates: list[Coordinate] | None = Field(
        default=None,
        description="Coordinates of the MDIO variable dimensions.",
    )
    metadata: VariableMetadata | None = Field(
        default=None, description="Variable metadata."
    )
