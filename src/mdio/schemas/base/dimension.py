"""Dimension schema."""

from pydantic import BaseModel
from pydantic import Field


class Dimension(BaseModel):
    """Data model defining a Dimension object."""

    name: str = Field(
        ...,
        description="Name of the dimension.",
    )
    size: int = Field(
        ...,
        gt=0,
        description="Size of the dimension.",
    )
    chunksize: int | None = Field(
        default=None,
        gt=0,
        description="Chunk size of the dimension.",
    )
