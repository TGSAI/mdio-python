"""Schemas for scalar types.

We take booleans, unsigned and signed integers, floats, and
complex numbers from numpy data types and allow those.
"""

from enum import StrEnum

import numpy as np
from pydantic import Field

from mdio.schemas.core import CamelCaseStrictModel


ALLOWED_TYPES = [
    np.sctypes["others"][0].__name__,  # boolean
    *[t.__name__ for t in np.sctypes["int"]],
    *[t.__name__ for t in np.sctypes["uint"]],
    *[t.__name__ for t in np.sctypes["float"]],
    *[t.__name__ for t in np.sctypes["complex"]],
]


ScalarType = StrEnum("ScalarType", {t.upper(): t for t in ALLOWED_TYPES})
ScalarType.__doc__ = """Scalar array data type."""


class StructuredField(CamelCaseStrictModel):
    """Structured array field with name, format, and byte offset."""

    format: ScalarType = Field()
    name: str | None = Field(default=None)
    offset: int | None = Field(default=None, ge=0)


class StructuredType(CamelCaseStrictModel):
    """Structured array type with fields and total item size."""

    fields: list[StructuredField] = Field()
    item_size: int | None = Field(default=None, gt=0)


class DataTypeModel(CamelCaseStrictModel):
    """Structured array type with fields and total item size."""

    data_type: ScalarType | StructuredType = Field(
        ..., description="Type of the array."
    )
