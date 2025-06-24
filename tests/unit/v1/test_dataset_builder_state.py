
from datetime import datetime
import pytest

from mdio.schemas.v1.dataset_builder import _BuilderState, MDIODatasetBuilder

def test_builder_initialization() -> None:
    """Test basic builder initialization."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder.name == "test_dataset"
    assert builder.api_version == "1.0.0"
    assert isinstance(builder.created_on, datetime)
    assert len(builder._dimensions) == 0
    assert len(builder._coordinates) == 0
    assert len(builder._variables) == 0
    assert builder._state == _BuilderState.INITIAL

def test_builder_add_dimension_state() -> None:
    """Test coordinate builder before and after add_dimension."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder._state == _BuilderState.INITIAL

    # One should be able to add dimension any time after the builder has been created

    # Add dimensions first
    builder = builder.add_dimension("x", 100)
    assert builder._state == _BuilderState.HAS_DIMENSIONS
    builder = builder.add_dimension("y", 200)
    assert builder._state == _BuilderState.HAS_DIMENSIONS

@pytest.mark.skip(reason="Under construction.")
def test_coordinate_builder_state() -> None:
    """Test coordinate builder state transitions and functionality."""
    builder = MDIODatasetBuilder("test_dataset")

    # Should not be able to add coordinates before dimensions
    with pytest.raises(
        ValueError, match="Must add at least one dimension before adding coordinates"
    ):
        builder.add_coordinate("x_coord", dimensions=["x"])

    # Add dimensions first
    builder = builder.add_dimension("x", 100)
    builder = builder.add_dimension("y", 200)

    # Adding coordinate should change state to HAS_COORDINATES
    builder = builder.add_coordinate("x_coord", dimensions=["x"], long_name="X Coordinate")
    assert builder._state == _BuilderState.HAS_COORDINATES
    assert len(builder._coordinates) == 1  # noqa: PLR2004
    assert builder._coordinates[0].name == "x_coord"
    assert builder._coordinates[0].long_name == "X Coordinate"
    assert builder._coordinates[0].dimensions[0].name == "x"

    # Adding another coordinate should maintain state
    builder = builder.add_coordinate("y_coord", dimensions=["y"])
    assert builder._state == _BuilderState.HAS_COORDINATES
    assert len(builder._coordinates) == 2  # noqa: PLR2004
    assert builder._coordinates[1].name == "y_coord"
    assert builder._coordinates[1].dimensions[0].name == "y"

@pytest.mark.skip(reason="Under construction.")
def test_variable_builder_state() -> None:
    """Test variable builder state transitions and functionality."""
    builder = MDIODatasetBuilder("test_dataset")

    # Should not be able to add variables before dimensions
    with pytest.raises(ValueError, match="Must add at least one dimension before adding variables"):
        builder.add_variable("data", dimensions=["x"])

    # Add dimension first
    builder = builder.add_dimension("x", 100)

    # Adding variable should change state to HAS_VARIABLES
    builder = builder.add_variable("data", dimensions=["x"], long_name="Data Variable")
    assert builder._state == _BuilderState.HAS_VARIABLES
    # One for dimension, one for variable
    assert len(builder._variables) == 2  # noqa: PLR2004
    assert builder._variables[1].name == "data"
    assert builder._variables[1].long_name == "Data Variable"
    assert builder._variables[1].dimensions[0].name == "x"

    # Adding another variable should maintain state
    builder = builder.add_variable("data2", dimensions=["x"])
    assert builder._state == _BuilderState.HAS_VARIABLES
    # One for dimension, two for variables
    assert len(builder._variables) == 3  # noqa: PLR2004
    assert builder._variables[2].name == "data2"
    assert builder._variables[2].dimensions[0].name == "x"