"""Common units for resource assessment data."""

from __future__ import annotations

from enum import Enum
from enum import unique

from pydantic import Field
from pydantic import create_model

from mdio.builder.schemas.core import CamelCaseStrictModel


@unique
class UnitEnum(str, Enum):
    """An Enum representing units as strings, from pint."""


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
