"""This module implements the core components of the MDIO schemas."""


from typing import Any
from typing import get_type_hints

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic.alias_generators import to_camel


class StrictCamelBaseModel(BaseModel):
    """A BaseModel subclass with Pascal Cased aliases."""

    model_config = ConfigDict(extra="forbid", alias_generator=to_camel)


class RegularChunkShape(StrictCamelBaseModel):
    """Represents regular chunk sizes along each dimension."""

    chunk_shape: list[int] = Field(
        ..., description="Lengths of the chunk along each dimension of the array."
    )


class RectilinearChunkShape(StrictCamelBaseModel):
    """Represents irregular chunk sizes along each dimension."""

    chunk_shape: list[list[int]] = Field(
        ...,
        description="Lengths of the chunk along each dimension of the array.",
    )


class RegularChunkGrid(StrictCamelBaseModel):
    """Represents a rectangular and regularly spaced chunk grid."""

    name: str = Field(default="regular", description="The name of the chunk grid.")

    configuration: RegularChunkShape = Field(
        ..., description="Configuration of the regular chunk grid."
    )


class RectilinearChunkGrid(StrictCamelBaseModel):
    """Represents a rectangular and irregularly spaced chunk grid."""

    name: str = Field(default="rectilinear", description="The name of the chunk grid.")

    configuration: RectilinearChunkShape = Field(
        ..., description="Configuration of the irregular chunk grid."
    )


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
