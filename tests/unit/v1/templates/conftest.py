"""Unit tests for the conftest module in the templates directory."""

# conftest.py
import pytest

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredField
from mdio.builder.schemas.dtype import StructuredType


@pytest.fixture(scope="session")
def structured_headers() -> StructuredType:
    """Fixture to provide structured headers for testing."""
    return StructuredType(
        fields=[
            StructuredField(name="cdp_x", format=ScalarType.INT32),
            StructuredField(name="cdp_y", format=ScalarType.INT32),
            StructuredField(name="elevation", format=ScalarType.FLOAT16),
            StructuredField(name="some_scalar", format=ScalarType.FLOAT16),
        ]
    )
