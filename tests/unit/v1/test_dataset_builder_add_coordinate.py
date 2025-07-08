# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 dataset_builder.add_coordinate() public API."""

import pytest

from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder, _get_named_dimension
from mdio.schemas.v1.dataset_builder import _BuilderState
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.variable import VariableMetadata


def test_add_coordinate() -> None:
    """Test adding coordinates. Check the state transition and validate required parameters."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder._state == _BuilderState.INITIAL
    
    msg = "Must add at least one dimension before adding coordinates"
    with pytest.raises(ValueError, match=msg):
        builder.add_coordinate("cdp", dimensions=["inline", "crossline"])

    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 200)

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
    # 1 variable for coordinates
    assert len(builder._variables) == 1 
    assert len(builder._coordinates) == 1
    
    # Validate that we created a coordinate variable
    var_cdp = next(e for e in builder._variables if e.name == "cdp-x")
    assert var_cdp is not None
    assert len(var_cdp.dimensions) == 2
    assert _get_named_dimension(var_cdp.dimensions, "inline", 100) is not None
    assert _get_named_dimension(var_cdp.dimensions, "crossline", 200) is not None
    # Validate that coordinates are stored as Coordinate
    assert len(var_cdp.coordinates) == 1
    # No dimensions are stored in coordinates
    # Validate that non-dimension coordinates
    assert builder._get_coordinate(var_cdp.coordinates, "cdp-x") is not None

    # Adding coordinate with the same name twice
    msg="Adding coordinate with the same name twice is not allowed"
    with pytest.raises(ValueError, match=msg):
        builder.add_coordinate("cdp-x", dimensions=["inline", "crossline"])

def test_add_coordinate_with_defaults() -> None:
    """Test adding coordinates with default arguments."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 200)
    # Add coordinate using defaults
    builder.add_coordinate("cdp", dimensions=["inline", "crossline"])
    assert len(builder._dimensions) == 2 
    # 1 variable for coordinates
    assert len(builder._variables) == 1
    assert len(builder._coordinates) == 1

    # Validate: the structure of the coordinate
    coord_cdp = next((e for e in builder._coordinates if e.name == "cdp"), None)
    assert coord_cdp is not None
    assert len(coord_cdp.dimensions) == 2
    assert _get_named_dimension(coord_cdp.dimensions, "inline", 100) is not None
    assert _get_named_dimension(coord_cdp.dimensions, "crossline", 200) is not None
    assert coord_cdp.long_name is None               # Default value
    assert coord_cdp.data_type == ScalarType.FLOAT32 # Default value
    assert coord_cdp.compressor is None              # Default value
    assert coord_cdp.metadata is None                # Default value


def test_coordinate_with_full_parameters() -> None:
    """Test adding coordinates with all metadata."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 200)
    # Add coordinate with all metadata
    builder.add_coordinate(
        "cdp",
        long_name = "Common Depth Point",
        dimensions=["inline", "crossline"],
        data_type = ScalarType.FLOAT16,
        compressor = Blosc(algorithm="zstd"),
        metadata_info=[
            AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT)),
            UserAttributes(attributes={"MGA": 51, "UnitSystem": "Imperial"})]
    )
    # Validate: the state of the builder
    assert builder._state == _BuilderState.HAS_COORDINATES
    assert len(builder._dimensions) == 2 
    # 1 variable for coordinates
    assert len(builder._variables) == 1 
    assert len(builder._coordinates) == 1 

    # Validate: the structure of the coordinate
    coord_cdp = next((e for e in builder._coordinates if e.name == "cdp"), None)
    assert coord_cdp is not None
    assert len(coord_cdp.dimensions) == 2
    assert _get_named_dimension(coord_cdp.dimensions, "inline", 100) is not None
    assert _get_named_dimension(coord_cdp.dimensions, "crossline", 200) is not None
    assert coord_cdp.long_name == "Common Depth Point"
    assert coord_cdp.data_type == ScalarType.FLOAT16 
    assert isinstance(coord_cdp.compressor, Blosc)
    assert coord_cdp.compressor.algorithm == "zstd"
    assert coord_cdp.metadata.attributes["MGA"] == 51
    assert coord_cdp.metadata.attributes["UnitSystem"] == "Imperial"
    assert coord_cdp.metadata.units_v1.length == LengthUnitEnum.FOOT

     # Validate: the structure of the created variable
    v = next((v for v in builder._variables if v.name == "cdp"), None)
    assert v is not None
    assert v.long_name == "'cdp' coordinate variable"
    assert len(v.dimensions) == 2
    assert _get_named_dimension(v.dimensions, "inline", 100) is not None
    assert _get_named_dimension(v.dimensions, "crossline", 200) is not None
    assert v.data_type == ScalarType.FLOAT16
    assert isinstance(v.compressor, Blosc)
    assert v.compressor.algorithm == "zstd"
    assert len(v.coordinates) == 1
    assert builder._get_coordinate(v.coordinates, "cdp") is not None
    assert isinstance(v.metadata, VariableMetadata)
    assert v.metadata.units_v1.length == LengthUnitEnum.FOOT
    assert v.metadata.attributes["MGA"] == 51
    assert v.metadata.attributes["UnitSystem"] == "Imperial"
