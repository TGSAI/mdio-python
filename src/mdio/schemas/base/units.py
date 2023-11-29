"""Common units for resource assessment data."""


from enum import Enum

from pydantic import Field

from mdio.schemas.base.core import StrictCamelBaseModel


class UnitEnum(str, Enum):
    """An Enum representing units as strings, from pint."""


def create_unit_model(
    unit_enum: type[UnitEnum],
    model_name: str,
    quantity: str,
) -> type[StrictCamelBaseModel]:
    """This generates a Pydantic BaseModel for a unit convention.

    Args:
        unit_enum: UnitEnum representing the units for a specific quantity.
        model_name: String representing the name of the unit model.
        quantity: String representing the quantity for which the unit model is created.

    Returns:
        A type representing the unit model derived from the BaseModel.

    Example:
        unit_enum = UnitEnum
        model_name = "LengthUnitModel"
        quantity = "length"
        create_unit_model(unit_enum, model_name, quantity)
    """
    attributes = {
        quantity: Field(..., description=f"Unit of {quantity}."),
        "__annotations__": {quantity: unit_enum},
    }

    # Construct the BaseModel
    unit_model = type(model_name, (StrictCamelBaseModel,), attributes)

    unit_model.__doc__ = f"Model representing units of {quantity}."

    return unit_model
