"""Tests the schema v1 dataset_builder internal methods."""

from datetime import UTC
from datetime import datetime

import pytest
from pydantic import Field

from mdio.schemas.core import StrictModel
from mdio.schemas.dimension import NamedDimension
from mdio.schemas.v1.dataset_builder import _get_named_dimension
from mdio.schemas.v1.dataset_builder import _to_dictionary


def test__get_named_dimension() -> None:
    """Test getting a dimension by name from the list of dimensions."""
    dimensions = [NamedDimension(name="inline", size=2), NamedDimension(name="crossline", size=3)]

    assert _get_named_dimension([], "inline") is None
    assert _get_named_dimension(dimensions, "inline") == NamedDimension(name="inline", size=2)
    assert _get_named_dimension(dimensions, "crossline") == NamedDimension(name="crossline", size=3)
    assert _get_named_dimension(dimensions, "time") is None

    with pytest.raises(TypeError, match="Expected str, got NoneType"):
        _get_named_dimension(dimensions, None)
    with pytest.raises(TypeError, match="Expected str, got int"):
        _get_named_dimension(dimensions, 42)
    with pytest.raises(
        ValueError, match="Dimension 'inline' found but size 2 does not match expected size 200"
    ):
        _get_named_dimension(dimensions, "inline", size=200)


def test__to_dictionary() -> None:
    """Test converting a dictionary, list or pydantic BaseModel to a dictionary."""
    # Validate inputs
    with pytest.raises(TypeError, match="Expected BaseModel, dict or list, got datetime"):
        _to_dictionary(datetime.now(UTC))

    # Convert None to None
    result = _to_dictionary(None)
    assert result is None

    # Validate conversion of a Pydantic BaseModel
    class SomeModel(StrictModel):
        count: int = Field(default=None, description="Samples count")
        samples: list[float] = Field(default_factory=list, description="Samples.")
        created: datetime = Field(
            default_factory=datetime.now, description="Creation time with TZ info."
        )

    md = SomeModel(
        count=3, samples=[1.0, 2.0, 3.0], created=datetime(2023, 10, 1, 12, 0, 0, tzinfo=UTC)
    )
    result = _to_dictionary(md)
    assert isinstance(result, dict)
    assert result == {"count": 3, "created": "2023-10-01T12:00:00Z", "samples": [1.0, 2.0, 3.0]}

    # Validate conversion of a dictionary
    dct = {
        "count": 3,
        "samples": [1.0, 2.0, 3.0],
        "created": datetime(2023, 10, 1, 12, 0, 0, tzinfo=UTC),
    }
    result = _to_dictionary(dct)
    assert isinstance(result, dict)
    assert result == {
        "count": 3,
        "samples": [1.0, 2.0, 3.0],
        "created": datetime(2023, 10, 1, 12, 0, 0, tzinfo=UTC),
    }

    # Validate conversion of a dictionary
    lst = [
        None,
        SomeModel(
            count=3, samples=[1.0, 2.0, 3.0], created=datetime(2023, 10, 1, 12, 0, 0, tzinfo=UTC)
        ),
        {
            "count2": 3,
            "samples2": [1.0, 2.0, 3.0],
            "created2": datetime(2023, 10, 1, 12, 0, 0, tzinfo=UTC),
        },
    ]
    result = _to_dictionary(lst)
    assert isinstance(result, dict)
    assert result == {
        "count": 3,
        "samples": [1.0, 2.0, 3.0],
        "created": "2023-10-01T12:00:00Z",
        "count2": 3,
        "samples2": [1.0, 2.0, 3.0],
        "created2": datetime(2023, 10, 1, 12, 0, 0, tzinfo=UTC),
    }
