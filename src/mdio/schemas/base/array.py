"""Base classes for Arrays."""


from pydantic import Field

from mdio.schemas.base import Compressors
from mdio.schemas.base.core import StrictCamelBaseModel
from mdio.schemas.base.dimension import DimensionContext


class BaseArray(StrictCamelBaseModel):
    """A base array schema."""

    dimensions: DimensionContext = Field(
        ..., description="List of Dimension collection or reference to dimension names."
    )
    compressor: Compressors | None = Field(
        default=None, description="Compression settings."
    )


class NamedArray(BaseArray):
    """An array with a name."""

    name: str = Field(..., description="Name of the array.")
    long_name: str | None = Field(default=None, description="Fully descriptive name.")
