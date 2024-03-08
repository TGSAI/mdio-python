"""Dimension schema."""

from pydantic import Field

from mdio.schemas.core import CamelCaseStrictModel


class NamedDimension(CamelCaseStrictModel):
    """Represents a single dimension with a name and size."""

    name: str = Field(..., description="Unique identifier for the dimension.")
    size: int = Field(..., gt=0, description="Total size of the dimension.")
