"""Common units for resource assessment data."""

from __future__ import annotations

from enum import StrEnum
from enum import unique
from typing import Self

from pydantic import Field
from pydantic import create_model

from mdio.builder.schemas.core import CamelCaseStrictModel


@unique
class UnitEnum(StrEnum):
    """An Enum representing units as strings, from pint."""

    def __new__(cls, value: object) -> Self:
        """Coerce Pint units to their configured string representation."""
        string_value = str(value)
        member = str.__new__(cls, string_value)
        member._value_ = string_value
        return member


def create_unit_model(
    unit_enum: type[UnitEnum],
    model_name: str,
    quantity: str,
    module: str,
) -> type[CamelCaseStrictModel]:
    """Dynamically creates a pydantic model from a unit Enum.

    Args:
        unit_enum: UnitEnum representing the units for a specific quantity.
        model_name: The name of the model to be created.
        quantity: String representing the quantity for which the unit model is created.
        module: Name of the module in which the model is to be created.
            This should be the `__name__` attribute of the module.

    Returns:
        A Pydantic Model representing the unit model derived from the BaseModel.

    Example:
        unit_enum = UnitEnum
        model_name = "LengthUnitModel"
        quantity = "length"
        create_unit_model(unit_enum, model_name, quantity)
    """
    fields = {quantity: (unit_enum, Field(..., description=f"Unit of {quantity}."))}

    return create_model(
        model_name,
        **fields,
        __base__=CamelCaseStrictModel,
        __doc__=f"Model representing units of {quantity}.",
        __module__=module,
    )
