"""This module implements the core components of the MDIO schemas."""


from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic.alias_generators import to_camel


class StrictCamelBaseModel(BaseModel):
    """A BaseModel subclass with Pascal Cased aliases."""

    model_config = ConfigDict(extra="forbid", alias_generator=to_camel)


class RegularChunkShape(StrictCamelBaseModel):
    """A data model for chunk sizes along each dimension."""

    chunk_shape: list[int] = Field(
        ..., description="Lengths of the chunk along each dimension of the array."
    )


class RectilinearChunkShape(StrictCamelBaseModel):
    """A data model for irregular chunk sizes along each dimension."""

    chunk_shape: list[list[int]] = Field(
        ...,
        description="Lengths of the chunk along each dimension of the array.",
    )


class RegularChunkGrid(StrictCamelBaseModel):
    """A data model representing a rectangular and regularly spaced chunk grid."""

    name: str = Field(default="regular", description="The name of the chunk grid.")

    configuration: RegularChunkShape = Field(
        ..., description="Configuration of the regular chunk grid."
    )


class RectilinearChunkGrid(StrictCamelBaseModel):
    """A data model representing a rectangular and irregularly spaced chunk grid."""

    name: str = Field(default="rectilinear", description="The name of the chunk grid.")

    configuration: RectilinearChunkShape = Field(
        ..., description="Configuration of the irregular chunk grid."
    )
