# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 dataset_builder.add_coordinate() public API."""

import pytest

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.chunk_grid import RegularChunkShape
from mdio.schemas.dimension import NamedDimension
from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder, contains_dimension
from mdio.schemas.v1.dataset_builder import _BuilderState
from mdio.schemas.v1.stats import CenteredBinHistogram
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.stats import SummaryStatistics
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel


def test_add_coordinate() -> None:
    """Test adding coordinates. Check the state transition and validate required parameters."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder._state == _BuilderState.INITIAL
    
    with pytest.raises(ValueError, match="Must add at least one dimension before adding coordinates"):
        builder.add_coordinate("amplitude", dimensions=["speed"])

    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)

    bad_name = None
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_coordinate(bad_name, dimensions=["speed"])
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_coordinate("", dimensions=["speed"])
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_coordinate("amplitude", dimensions=None)
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_coordinate("amplitude", dimensions=[])

    builder.add_coordinate("cdp", dimensions=["inline", "crossline"])
    assert builder._state == _BuilderState.HAS_COORDINATES
    assert len(builder._dimensions) == 2 
    assert len(builder._variables) == 2 
    assert len(builder._coordinates) == 1  

def test_add_coordinate_with_defaults() -> None:
    """Test adding coordinates with default arguments."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)
    # Add coordinate using defaults
    builder.add_coordinate("cdp", dimensions=["inline", "crossline"])
    assert len(builder._dimensions) == 2 
    assert len(builder._variables) == 2 
    assert len(builder._coordinates) == 1
    # NOTE: add_variable() stores dimensions as names
    crd0 = next((e for e in builder._coordinates if e.name == "cdp"), None)
    assert crd0 is not None
    assert set(crd0.dimensions) == set(["inline", "crossline"])
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
    assert len(builder._variables) == 2  
    assert len(builder._coordinates) == 1 
    # NOTE: add_coordinate() stores dimensions as names
    crd0 = next((e for e in builder._coordinates if e.name == "cdp"), None)
    assert crd0 is not None
    assert set(crd0.dimensions) == set(["inline", "crossline"])
    assert crd0.long_name is None               # Default value
    assert crd0.data_type == ScalarType.FLOAT32 # Default value
    assert crd0.metadata.attributes is None
    assert crd0.metadata.units_v1.length == "ft"


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
    assert len(builder._variables) == 2  
    assert len(builder._coordinates) == 1 
    # NOTE: add_coordinate() stores dimensions as names
    crd0 = next((e for e in builder._coordinates if e.name == "cdp"), None)
    assert crd0 is not None
    assert set(crd0.dimensions) == set(["inline", "crossline"])
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
    assert len(builder._variables) == 2  
    assert len(builder._coordinates) == 1 
    # NOTE: add_coordinate() stores dimensions as names
    crd0 = next((e for e in builder._coordinates if e.name == "cdp"), None)
    assert crd0 is not None
    assert set(crd0.dimensions) == set(["inline", "crossline"])
    assert crd0.long_name is None               # Default value
    assert crd0.data_type == ScalarType.FLOAT32 # Default value
    assert crd0.metadata.attributes["MGA"] == 51
    assert crd0.metadata.attributes["UnitSystem"] == "Imperial"
    assert crd0.metadata.units_v1.length == "ft"

