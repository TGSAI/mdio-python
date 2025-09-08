"""Dimension tests."""

import pytest

from mdio.core import Dimension
from mdio.exceptions import ShapeError


@pytest.fixture
def my_dimension() -> Dimension:
    """Mock dimension."""
    return Dimension(coords=range(10, 18, 2), name="dim_0")


class TestDimension:
    """Basic tests for reading or manipulating dimensions."""

    def test_len(self, my_dimension: Dimension) -> None:
        """Test length method."""
        assert len(my_dimension) == 4

    @pytest.mark.parametrize(("index", "expected"), [(1, 12), (-1, 16), (2, 14)])
    def test_getitem(self, my_dimension: Dimension, index: int, expected: int) -> None:
        """Test getter (integer indexing)."""
        assert my_dimension[index] == expected

    @pytest.mark.parametrize(("index", "expected"), [(1, 12), (-1, 16), (2, 14)])
    def test_setitem(self, index: int, expected: int) -> None:
        """Test setter (integer indexing)."""
        other_dim = Dimension(coords=range(4), name="dim_6")
        other_dim[index] = expected
        assert other_dim[index] == expected

    def test_hash_equality(self, my_dimension: Dimension) -> None:
        """Test hashing (and equality checks)."""
        other_dim1 = Dimension(coords=range(10, 18, 2), name="dim_0")
        other_dim2 = Dimension(coords=range(15), name="dim_1")
        assert my_dimension == other_dim1
        assert my_dimension != other_dim2


class TestExceptions:
    """Test custom exceptions and if they're raised properly."""

    def test_shape_error(self) -> None:
        """Wrong shape."""
        with pytest.raises(ShapeError):
            Dimension(coords=[range(10, 18, 2)] * 2, name="dim_0")

    def test_wrong_type_equals(self, my_dimension: Dimension) -> None:
        """Wrong type."""
        with pytest.raises(TypeError):
            assert my_dimension == ("not", "a", "Dimension")
