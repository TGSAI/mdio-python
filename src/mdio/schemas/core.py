"""This module implements the core components of the MDIO schemas."""

from __future__ import annotations

from typing import Any
from typing import get_type_hints

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel
from pydantic import Field


# def model_fields(model: type[BaseModel]) -> dict[str, tuple[Any, Any]]:
#     """Extract Pydantic BaseModel fields.

#     Args:
#         model: (Type) The model object for which the fields will be extracted.

#     Returns:
#         A dictionary containing the fields of the model along with
#         their corresponding types and default values.

#     Example:
#         >>> class MyModel(BaseModel):
#         ...     name: str
#         ...     age: int = 0
#         ...
#         >>> model_fields(MyModel)
#         {'name': (str, <default_value>), 'age': (int, 0)}
#     """
#     annotations = get_type_hints(model)

#     fields = {}
#     for field_name, field in model.model_fields.items():
#         fields[field_name] = (annotations[field_name], field)

#     return fields

# def model_fields(model: type[BaseModel]) -> dict[str, tuple[Any, Any]]:
#     """Return fields suitable for use in create_model with correct types and defaults."""
#     fields = {}
#     for field_name, field_info in model.model_fields.items():
#         annotated_type = field_info.annotation
#         default = field_info.default if field_info.default is not None else ...
#         fields[field_name] = (annotated_type, Field(default, description=field_info.description))
#     return fields

def model_fields(model: type[BaseModel]) -> dict[str, tuple[Any, Any]]:
    """Safely extract fields for create_model, preserving optionality and default behavior."""
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

    def model_dump_json(self, *args, **kwargs):  # type: ignore[override]
        """Dump JSON using camelCase aliases and excluding None values by default."""
        # Ensure camelCase aliases
        if "by_alias" not in kwargs:
            kwargs["by_alias"] = True
        # Exclude None fields to avoid nulls in output
        if "exclude_none" not in kwargs:
            kwargs["exclude_none"] = True
        return super().model_dump_json(*args, **kwargs)
