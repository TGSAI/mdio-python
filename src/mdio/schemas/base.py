"""Base models to subclass from."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ConfigDict
from pydantic import Field
from pydantic.json_schema import GenerateJsonSchema

from mdio.schemas.core import CamelCaseStrictModel
from mdio.schemas.dtype import DataTypeModel


if TYPE_CHECKING:
    from mdio.schemas.compressors import ZFP
    from mdio.schemas.compressors import Blosc
    from mdio.schemas.dimension import NamedDimension


JSON_SCHEMA_DIALECT = GenerateJsonSchema.schema_dialect


class BaseDataset(CamelCaseStrictModel):
    """A base class for MDIO datasets.

    We add schema dialect to extend the config of `StrictCamelBaseModel`.
    We use the default Pydantic schema generator `GenerateJsonSchema` to
    define the JSON schema dialect accurately.
    """

    model_config = ConfigDict(json_schema_extra={"$schema": JSON_SCHEMA_DIALECT})


class BaseArray(DataTypeModel, CamelCaseStrictModel):
    """A base array schema."""

    dimensions: list[NamedDimension] | list[str] = Field(
        ..., description="List of Dimension collection or reference to dimension names."
    )
    compressor: Blosc | ZFP | None = Field(
        default=None, description="Compression settings."
    )


class NamedArray(BaseArray):
    """An array with a name."""

    name: str = Field(..., description="Name of the array.")
    long_name: str | None = Field(default=None, description="Fully descriptive name.")
