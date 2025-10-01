"""Schemas for scalar types.

We take booleans, unsigned and signed integers, floats, and
complex numbers from numpy data types and allow those.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from mdio.builder.schemas.core import CamelCaseStrictModel


class ScalarType(StrEnum):
    """Scalar array data type."""

    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    FLOAT128 = "float128"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    COMPLEX256 = "complex256"
    BYTES240 = "V240"  # fixed-width 240-byte string, used for raw v0/1/2 trace headers


class StructuredField(CamelCaseStrictModel):
    """Structured array field with name, format."""

    format: ScalarType = Field(...)
    name: str = Field(...)


class StructuredType(CamelCaseStrictModel):
    """Structured array type with packed fields."""

    fields: list[StructuredField] = Field()


class DataTypeModel(CamelCaseStrictModel):
    """Structured array type with fields and total item size."""

    data_type: ScalarType | StructuredType = Field(..., description="Type of the array.")
