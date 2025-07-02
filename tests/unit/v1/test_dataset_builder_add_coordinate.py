# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 dataset_builder.add_coordinate() public API."""

import pytest

from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.dataset_builder import _BuilderState
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel


def test_add_coordinate() -> None:
    """Test adding coordinates. Check the state transition and validate required parameters."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder._state == _BuilderState.INITIAL
    
    msg = "Must add at least one dimension before adding coordinates"
    with pytest.raises(ValueError, match=msg):
        builder.add_coordinate("cdp", dimensions=["inline", "crossline"])

    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)

    # Validate required parameters
    bad_name = None
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_coordinate(bad_name, dimensions=["speed"])
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_coordinate("", dimensions=["speed"])
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_coordinate("cdp-x", dimensions=None)
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_coordinate("cdp-x", dimensions=[])

    # Add a variable using non-existent dimensions
    msg="Pre-existing dimension named 'xline' is not found"
    with pytest.raises(ValueError, match=msg):
        builder.add_coordinate("bad_cdp-x", dimensions=["inline", "xline"])

    # Validate state transition
    builder.add_coordinate("cdp-x", dimensions=["inline", "crossline"])
    assert builder._state == _BuilderState.HAS_COORDINATES
    assert len(builder._dimensions) == 2 
    # 2 variables for dimensions, 1 variable for coordinates
    assert len(builder._variables) == 3 
    assert len(builder._coordinates) == 1
    
    # Validate that we created a coordinate variable
    var_cdp = next(e for e in builder._variables if e.name == "cdp-x")
    assert var_cdp is not None
    # Validate that dimensions are stored as names
    assert set(var_cdp.dimensions) == {"inline", "crossline"}
    # Validate that coordinates are stored as Coordinate
    assert len(var_cdp.coordinates) == 1
    assert next((e for e in var_cdp.coordinates if e.name == "cdp-x"), None) is not None

    # Adding coordinate with the same name twice
    msg="Adding coordinate with the same name twice is not allowed"
    with pytest.raises(ValueError, match=msg):
        builder.add_coordinate("cdp-x", dimensions=["inline", "crossline"])

def test_add_coordinate_with_defaults() -> None:
    """Test adding coordinates with default arguments."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)
    # Add coordinate using defaults
    builder.add_coordinate("cdp", dimensions=["inline", "crossline"])
    assert len(builder._dimensions) == 2 
    # 2 variables for dimensions, 1 variable for coordinates
    assert len(builder._variables) == 3
    assert len(builder._coordinates) == 1
    crd0 = next((e for e in builder._coordinates if e.name == "cdp"), None)
    assert crd0 is not None
    # NOTE: add_variable() stores dimensions as names
    assert set(crd0.dimensions) == {"inline", "crossline"}
    assert crd0.long_name is None               # Default value
    assert crd0.data_type == ScalarType.FLOAT32 # Default value
    assert crd0.metadata is None                # Default value
 
def test_coordinate_with_units() -> None:
    """Test adding coordinates with units."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)
    # Add coordinate with units
    builder.add_coordinate(
        "cdp", 
        dimensions=["inline", "crossline"],
        metadata_info=[AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT))]
    )
    assert len(builder._dimensions) == 2 
    # 2 variables for dimensions, 1 variable for coordinates
    assert len(builder._variables) == 3  
    assert len(builder._coordinates) == 1 
    crd0 = next((e for e in builder._coordinates if e.name == "cdp"), None)
    assert crd0 is not None
    # NOTE: add_coordinate() stores dimensions as names
    assert set(crd0.dimensions) == {"inline", "crossline"}
    assert crd0.long_name is None               # Default value
    assert crd0.data_type == ScalarType.FLOAT32 # Default value
    assert crd0.metadata.attributes is None
    assert crd0.metadata.units_v1.length == LengthUnitEnum.FOOT


def test_coordinate_with_attributes() -> None:
    """Test adding coordinates with attributes."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)
    # Add coordinate with attributes
    builder.add_coordinate(
        "cdp", 
        dimensions=["inline", "crossline"], 
        metadata_info=[UserAttributes(attributes={"MGA": 51, "UnitSystem": "Imperial"})],
    )
    assert len(builder._dimensions) == 2 
    # 2 variables for dimensions, 1 variable for coordinates
    assert len(builder._variables) == 3  
    assert len(builder._coordinates) == 1 
    # NOTE: add_coordinate() stores dimensions as names
    crd0 = next((e for e in builder._coordinates if e.name == "cdp"), None)
    assert crd0 is not None
    assert set(crd0.dimensions) == {"inline", "crossline"}
    assert crd0.long_name is None               # Default value
    assert crd0.data_type == ScalarType.FLOAT32 # Default value
    assert crd0.metadata.attributes["MGA"] == 51
    assert crd0.metadata.attributes["UnitSystem"] == "Imperial"
    assert crd0.metadata.units_v1 is None


def test_coordinate_with_full_metadata() -> None:
    """Test adding coordinates with all metadata."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)
    # Add coordinate with all metadata
    builder.add_coordinate(
        "cdp",
        dimensions=["inline", "crossline"],
        metadata_info=[
            AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT)),
            UserAttributes(attributes={"MGA": 51, "UnitSystem": "Imperial"})]
    )
    assert len(builder._dimensions) == 2 
    # 2 variables for dimensions, 1 variable for coordinates
    assert len(builder._variables) == 3  
    assert len(builder._coordinates) == 1 
    # NOTE: add_coordinate() stores dimensions as names
    crd0 = next((e for e in builder._coordinates if e.name == "cdp"), None)
    assert crd0 is not None
    assert set(crd0.dimensions) == {"inline", "crossline"}
    assert crd0.long_name is None               # Default value
    assert crd0.data_type == ScalarType.FLOAT32 # Default value
    assert crd0.metadata.attributes["MGA"] == 51
    assert crd0.metadata.attributes["UnitSystem"] == "Imperial"
    assert crd0.metadata.units_v1.length == LengthUnitEnum.FOOT

