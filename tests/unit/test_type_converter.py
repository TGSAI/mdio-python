"""Unit tests for the type converter module."""

import numpy as np
import pytest
from pydantic import ValidationError

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredField
from mdio.builder.schemas.dtype import StructuredType
from mdio.converters.type_converter import to_numpy_dtype
from mdio.converters.type_converter import to_scalar_type
from mdio.converters.type_converter import to_structured_type


@pytest.fixture
def supported_scalar_types_map() -> tuple[tuple[ScalarType, str], ...]:
    """Supported scalar types and their numpy equivalents."""
    return (
        (ScalarType.INT8, "int8"),
        (ScalarType.INT16, "int16"),
        (ScalarType.INT32, "int32"),
        (ScalarType.INT64, "int64"),
        (ScalarType.UINT8, "uint8"),
        (ScalarType.UINT16, "uint16"),
        (ScalarType.UINT32, "uint32"),
        (ScalarType.UINT64, "uint64"),
        (ScalarType.FLOAT32, "float32"),
        (ScalarType.FLOAT64, "float64"),
        (ScalarType.COMPLEX64, "complex64"),
        (ScalarType.COMPLEX128, "complex128"),
        (ScalarType.BOOL, "bool"),
    )


@pytest.fixture
def a_structured_type() -> StructuredType:
    """Sample structured type.

    Returns a structured type.
    """
    return StructuredType(
        fields=[
            StructuredField(name="x", format=ScalarType.FLOAT64),
            StructuredField(name="y", format=ScalarType.FLOAT64),
            StructuredField(name="z", format=ScalarType.FLOAT64),
            StructuredField(name="id", format=ScalarType.INT32),
            StructuredField(name="valid", format=ScalarType.BOOL),
        ]
    )


def test_to_numpy_dtype(supported_scalar_types_map: tuple[ScalarType, str], a_structured_type: StructuredType) -> None:
    """Comprehensive test for to_numpy_dtype function."""
    # Test 0: invalid input
    err = "Expected ScalarType, FixedStringType, or StructuredType, got 'str'"
    with pytest.raises(ValueError, match=err):
        to_numpy_dtype("parameter of invalid type")

    # Test 1: ScalarType cases - all supported scalar types
    for scalar_type, expected_numpy_type in supported_scalar_types_map:
        result = to_numpy_dtype(scalar_type)
        expected = np.dtype(expected_numpy_type)
        assert result == expected
        assert isinstance(result, np.dtype)
        assert result.name == expected.name

    # Test 2: StructuredType with multiple fields
    result_multi = to_numpy_dtype(a_structured_type)
    expected_multi = np.dtype(
        [("x", "float64"), ("y", "float64"), ("z", "float64"), ("id", "int32"), ("valid", "bool")]
    )

    assert result_multi == expected_multi
    assert isinstance(result_multi, np.dtype)
    assert len(result_multi.names) == 5
    assert set(result_multi.names) == {"x", "y", "z", "id", "valid"}


@pytest.mark.parametrize("token", ["S1", "S8", "S32", "U1", "U8"])
def test_fixed_string_roundtrip(token: str) -> None:
    """Fixed-length byte and unicode strings survive conversion at any width."""
    dtype = np.dtype(token)
    assert to_scalar_type(dtype) == token
    assert to_numpy_dtype(token) == np.dtype(token)

    structured = to_structured_type(np.dtype([("label", token), ("inline", "int32")]))
    assert structured.fields[0].format == token
    assert to_numpy_dtype(structured).fields["label"][0] == np.dtype(token)


@pytest.mark.parametrize("token", ["S0", "U0", "bytes64", "str256", "V8", "|S8"])
def test_structured_field_rejects_noncanonical_string(token: str) -> None:
    """Schema accepts only canonical S<n> and U<n> tokens."""
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        StructuredField(name="label", format=token)


def test_fixed_string_ignores_numpy_display_name() -> None:
    """Numpy's bytes64 / str256 names are not valid dtype constructors."""
    assert to_scalar_type(np.dtype("|S8")) == "S8"
    assert to_scalar_type(np.dtype(">U8")) == "U8"
    with pytest.raises(ValueError, match="void64"):
        to_scalar_type(np.dtype("V8"))


def test_to_scalar_type(supported_scalar_types_map: tuple[ScalarType, str]) -> None:
    """Test for to_scalar_type function."""
    for expected_mdio_type, numpy_type in supported_scalar_types_map:
        result = to_scalar_type(np.dtype(numpy_type))
        assert result == expected_mdio_type


def test_to_structured_type(a_structured_type: StructuredType) -> None:
    """Test for to_structured_type function."""
    dtype = np.dtype([("x", "float64"), ("y", "float64"), ("z", "float64"), ("id", "int32"), ("valid", "bool")])
    assert a_structured_type == to_structured_type(dtype)

    dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("id", "<i4"), ("valid", "?")])
    assert a_structured_type == to_structured_type(dtype)
