"""Tests the schema v1 dataset_builder internal methods."""

import pytest

from mdio.builder.dataset_builder import _get_named_dimension
from mdio.builder.schemas.dimension import NamedDimension


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
    with pytest.raises(ValueError, match="Dimension 'inline' found but size 2 does not match expected size 200"):
        _get_named_dimension(dimensions, "inline", size=200)
