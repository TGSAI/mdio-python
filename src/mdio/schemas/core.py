"""This module implements the core components of the MDIO schemas."""

from __future__ import annotations

from typing import Any
from typing import get_type_hints

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel


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


class StrictModel(BaseModel):
    """A model with forbidden extras."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CamelCaseStrictModel(StrictModel):
    """A model with forbidden extras and camel case aliases."""

    model_config = ConfigDict(alias_generator=to_camel)
