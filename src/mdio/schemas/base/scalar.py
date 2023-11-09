"""Schemas for scalar types.

We take booleans, unsigned and signed integers, floats, and
complex numbers from numpy data types and allow those.
"""


from enum import StrEnum

import numpy as np
from pydantic import BaseModel
from pydantic import Field


ALLOWED_TYPES = [
    np.sctypes["others"][0].__name__,  # boolean
    *[t.__name__ for t in np.sctypes["int"]],
    *[t.__name__ for t in np.sctypes["uint"]],
    *[t.__name__ for t in np.sctypes["float"]],
    *[t.__name__ for t in np.sctypes["complex"]],
]


ScalarType = StrEnum("ScalarType", {t.upper(): t for t in ALLOWED_TYPES})


class DataType(BaseModel):
    """Represents an array type with a specific format and byte order."""

    format: ScalarType = Field()
    name: str | None = Field(default=None)
    offset: int | None = Field(default=None, ge=0)


class StructuredDataType(BaseModel):
    """Structured array type with field names, formats, offsets, and item size."""

    formats: list[DataType] = Field()
    item_size: int | None = Field(default=None, gt=0)
