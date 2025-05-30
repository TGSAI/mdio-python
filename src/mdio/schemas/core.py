"""This module implements the core components of the MDIO schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
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
    fields = {}
    for field_name, field_info in model.model_fields.items():
        annotated_type = field_info.annotation
        if field_info.is_required():
            fields[field_name] = (annotated_type, ...)
        else:
            fields[field_name] = (
                annotated_type,
                Field(field_info.default, description=field_info.description),
            )
    return fields


class StrictModel(BaseModel):
    """A model with forbidden extras."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CamelCaseStrictModel(StrictModel):
    """A model with forbidden extras and camel case aliases."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=False,
        alias_generator=to_camel,
        ser_json_by_alias=True,
    )

    def model_dump_json(self, *args, **kwargs) -> dict:  # noqa: ANN201 ANN001 ANN002 ANN003
        """Dump JSON using camelCase aliases and excluding None values by default."""
        # Ensure camelCase aliases
        if "by_alias" not in kwargs:
            kwargs["by_alias"] = True
        # Exclude None fields to avoid nulls in output
        if "exclude_none" not in kwargs:
            kwargs["exclude_none"] = True
        return super().model_dump_json(*args, **kwargs)

    def json(self, *args, **kwargs) -> dict:  # noqa: ANN201 ANN001 ANN002 ANN003
        """Dump JSON using camelCase aliases and excluding None values by default."""
        if "by_alias" not in kwargs:
            kwargs["by_alias"] = True
        return self.model_dump_json(*args, **kwargs)
