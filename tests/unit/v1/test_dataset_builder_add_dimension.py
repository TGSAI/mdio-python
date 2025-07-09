# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 dataset_builder.add_dimension() public API."""

import pytest

from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.dataset_builder import _BuilderState
from mdio.schemas.v1.dataset_builder import _get_named_dimension

from .helpers import validate_builder


def test_add_dimension() -> None:
    """Test adding dimension. Check the state transition and validate required parameters."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder._state == _BuilderState.INITIAL

    # Validate required parameters
    bad_name = None
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_dimension(bad_name, 200)
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_dimension("", 200)

    # First dimension should change state to HAS_DIMENSIONS and create a variable
    builder.add_dimension("x", 100)
    validate_builder(builder, _BuilderState.HAS_DIMENSIONS, n_dims=1, n_coords=0, n_var=0)
    assert _get_named_dimension(builder._dimensions, "x", 100) is not None

    # Validate that we can't add a dimension with the same name twice
    with pytest.raises(
        ValueError,
        match="Adding dimension with the same name twice is not allowed",
    ):
        builder.add_dimension("x", 200)

    # Adding dimension with the same name twice
    msg="Adding dimension with the same name twice is not allowed"
    with pytest.raises(ValueError, match=msg):
        builder.add_dimension("x", 200)

