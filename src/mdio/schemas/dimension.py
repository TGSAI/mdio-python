"""Dimension schema."""


from typing import TypeAlias

from pydantic import Field

from mdio.schemas.base.core import StrictCamelBaseModel


class Dimension(StrictCamelBaseModel):
    """Represents a single dimension with a name and size."""

    name: str = Field(..., description="Unique identifier for the dimension.")
    size: int = Field(..., gt=0, description="Total size of the dimension.")


DimensionCollection: TypeAlias = list[Dimension]
DimensionReference: TypeAlias = list[str]

DimensionContext: TypeAlias = list[Dimension] | list[str]
