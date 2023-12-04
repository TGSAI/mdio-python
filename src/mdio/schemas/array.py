"""Base classes for Arrays."""


from pydantic import Field

from mdio.schemas.base.core import StrictCamelBaseModel
from mdio.schemas.compressors import ZFP
from mdio.schemas.compressors import Blosc
from mdio.schemas.dimension import Dimension


class BaseArray(StrictCamelBaseModel):
    """A base array schema."""

    dimensions: list[Dimension] | list[str] = Field(
        ..., description="List of Dimension collection or reference to dimension names."
    )
    compressor: Blosc | ZFP | None = Field(
        default=None, description="Compression settings."
    )


class NamedArray(BaseArray):
    """An array with a name."""

    name: str = Field(..., description="Name of the array.")
    long_name: str | None = Field(default=None, description="Fully descriptive name.")
