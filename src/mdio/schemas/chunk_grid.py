"""This module contains data models for Zarr's chunk grid."""

from pydantic import Field

from mdio.schemas.core import CamelCaseStrictModel


class RegularChunkShape(CamelCaseStrictModel):
    """Represents regular chunk sizes along each dimension."""

    chunk_shape: list[int] = Field(
        ..., description="Lengths of the chunk along each dimension of the array."
    )


class RectilinearChunkShape(CamelCaseStrictModel):
    """Represents irregular chunk sizes along each dimension."""

    chunk_shape: list[list[int]] = Field(
        ...,
        description="Lengths of the chunk along each dimension of the array.",
    )


class RegularChunkGrid(CamelCaseStrictModel):
    """Represents a rectangular and regularly spaced chunk grid."""

    name: str = Field(default="regular", description="The name of the chunk grid.")

    configuration: RegularChunkShape = Field(
        ..., description="Configuration of the regular chunk grid."
    )


class RectilinearChunkGrid(CamelCaseStrictModel):
    """Represents a rectangular and irregularly spaced chunk grid."""

    name: str = Field(default="rectilinear", description="The name of the chunk grid.")

    configuration: RectilinearChunkShape = Field(
        ..., description="Configuration of the irregular chunk grid."
    )
