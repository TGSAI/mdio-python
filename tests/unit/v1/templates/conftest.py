# conftest.py
import pytest

from mdio.schemas.dtype import ScalarType, StructuredField, StructuredType

@pytest.fixture(scope="session")
def structured_headers():
    """Fixture to provide structured headers for testing."""
    headers = StructuredType(
        fields=[
            StructuredField(name="cdp_x", format=ScalarType.INT32),
            StructuredField(name="cdp_y", format=ScalarType.INT32),
            StructuredField(name="elevation", format=ScalarType.FLOAT16),
            StructuredField(name="some_scalar", format=ScalarType.FLOAT16),
        ]
    )
    return headers