"""Dimension schema."""

from pydantic import BaseModel
from pydantic import Field


class Dimension(BaseModel):
    """Data model for a dimension."""

    name: str = Field(description="Name of the dimension.")
    size: int = Field(description="Size of the dimension.")
