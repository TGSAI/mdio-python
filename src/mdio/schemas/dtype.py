"""Schemas for scalar types.

We take booleans, unsigned and signed integers, floats, and
complex numbers from numpy data types and allow those.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import Field

from mdio.schemas.core import CamelCaseStrictModel


ALLOWED_TYPES = [
    np.sctypes["others"][0].__name__,  # boolean # noqa: NPY201
    *[t.__name__ for t in np.sctypes["int"]],  # noqa: NPY201
    *[t.__name__ for t in np.sctypes["uint"]],  # noqa: NPY201
    *[t.__name__ for t in np.sctypes["float"]],  # noqa: NPY201
    *[t.__name__ for t in np.sctypes["complex"]],  # noqa: NPY201
]


ScalarType = StrEnum("ScalarType", {t.upper(): t for t in ALLOWED_TYPES})
ScalarType.__doc__ = """Scalar array data type."""


class StructuredField(CamelCaseStrictModel):
    """Structured array field with name, format."""

    format: ScalarType = Field(...)
    name: str = Field(...)


class StructuredType(CamelCaseStrictModel):
    """Structured array type with packed fields."""

    fields: list[StructuredField] = Field()


class DataTypeModel(CamelCaseStrictModel):
    """Structured array type with fields and total item size."""

    data_type: ScalarType | StructuredType = Field(
        ..., description="Type of the array."
    )
