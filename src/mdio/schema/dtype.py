"""Schemas for scalar types.

We take booleans, unsigned and signed integers, floats, and
complex numbers from numpy data types and allow those.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import Field

from mdio.schema.core import CamelCaseStrictModel


ALLOWED_TYPES = [
    # Boolean
    np.bool_.__name__,
    # Signed integers
    np.int8.__name__,
    np.int16.__name__,
    np.int32.__name__,
    np.int64.__name__,
    # Unsigned integers
    np.uint8.__name__,
    np.uint16.__name__,
    np.uint32.__name__,
    np.uint64.__name__,
    # Floating point
    np.float16.__name__,
    np.float32.__name__,
    np.float64.__name__,
    np.float128.__name__,
    # Complex
    np.complex64.__name__,
    np.complex128.__name__,
    np.clongdouble.__name__,
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
