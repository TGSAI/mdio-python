# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 Variable public API."""

import pytest

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.chunk_grid import RegularChunkShape
from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.dataset_builder import _BuilderState
from mdio.schemas.v1.stats import CenteredBinHistogram
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.stats import SummaryStatistics
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.variable import VariableMetadata


def test_add_variable() -> None:
    """Test adding variable. Check the state transition and validate required parameters.."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder._state == _BuilderState.INITIAL
    
    msg = "Must add at least one dimension before adding variables"
    with pytest.raises(ValueError, match=msg):
        builder.add_variable("amplitude", dimensions=["speed"], data_type = ScalarType.FLOAT32)

    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)
    builder.add_dimension("depth", 100)

    # Validate required parameters
    bad_name = None
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_variable(bad_name, dimensions=["speed"], data_type = ScalarType.FLOAT32)
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_variable("", dimensions=["speed"], data_type = ScalarType.FLOAT32)
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_variable("bad_amplitude", dimensions=None, data_type = ScalarType.FLOAT32)
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_variable("bad_amplitude", dimensions=[], data_type = ScalarType.FLOAT32)

    # Add a variable using non-existent dimensions
    msg="Pre-existing dimension named 'xline' is not found"
    with pytest.raises(ValueError, match=msg):
        builder.add_variable("bad_amplitude", 
                             dimensions=["inline", "xline", "depth"], 
                             data_type = ScalarType.FLOAT32)

    builder.add_variable("amplitude", 
                         dimensions=["inline", "crossline", "depth"], 
                         data_type = ScalarType.FLOAT32)
    assert builder._state == _BuilderState.HAS_VARIABLES
    assert len(builder._dimensions) == 3 
    assert len(builder._variables) == 4 
    assert len(builder._coordinates) == 0  

    # Add a variable using non-existent coordinates
    msg="Pre-existing coordinate named 'cdp-x' is not found"
    with pytest.raises(ValueError, match=msg):
        builder.add_variable("bad_amplitude", 
                             dimensions=["inline", "crossline", "depth"],
                             data_type = ScalarType.FLOAT32,
                             coordinates=["cdp-x", "cdp-y"])    

    builder.add_coordinate("cdp-x", dimensions=["inline", "crossline"])
    builder.add_coordinate("cdp-y", dimensions=["inline", "crossline"])

    # Adding variable with the same name twice
    msg="Adding variable with the same name twice is not allowed"
    with pytest.raises(ValueError, match=msg):
        builder.add_variable("amplitude", 
                             dimensions=["inline", "crossline", "depth"],
                             data_type = ScalarType.FLOAT32)


def test_add_variable_with_defaults() -> None:
    """Test adding variable with default arguments."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)
    builder.add_dimension("depth", 100)
    # Add variable using defaults
    builder.add_variable("seismic_amplitude", 
                         dimensions=["inline", "crossline", "depth"],
                         data_type=ScalarType.FLOAT32)
    assert len(builder._dimensions) == 3 
    assert len(builder._variables) == 4 
    assert len(builder._coordinates) == 0 
    var0 = next((e for e in builder._variables if e.name == "seismic_amplitude"), None)
    assert var0 is not None
    # NOTE: add_variable() stores dimensions as names
    assert set(var0.dimensions) == {"inline", "crossline", "depth"}
    assert var0.long_name is None               # Default value
    assert var0.data_type == ScalarType.FLOAT32 # Default value
    assert var0.compressor is None              # Default value
    assert var0.coordinates is None             # Default value
    assert var0.metadata is None                # Default value

def test_add_variable_full_parameters() -> None:
    """Test adding variable with full parameters."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)
    builder.add_dimension("depth", 100)
    builder.add_coordinate("cdp-x", dimensions=["inline", "crossline"])
    builder.add_coordinate("cdp-y", dimensions=["inline", "crossline"])
    builder.add_variable("seismic_amplitude", 
        long_name="Amplitude (dimensionless)",
        dimensions=["inline", "crossline", "depth"],    
        data_type=ScalarType.FLOAT64, 
        compressor=Blosc(algorithm="zstd"), 
        coordinates=["cdp-x", "cdp-y"],
        metadata_info=[
            AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT)),
            UserAttributes(attributes={"MGA": 51, "UnitSystem": "Imperial"}),
            ChunkGridMetadata(
                chunk_grid=RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=[20]))
            ),
            StatisticsMetadata(
                stats_v1=SummaryStatistics(
                    count=100,
                    sum=1215.1,
                    sumSquares=125.12,
                    min=5.61,
                    max=10.84,
                    histogram=CenteredBinHistogram(binCenters=[1, 2], counts=[10, 15]),
                )
            ),
        ])
    assert len(builder._dimensions) == 3
    assert len(builder._coordinates) == 2  
    # We expect 6 variables:
    # 3 variables for dimensions, 2 variables for coordinates, and 1 variable for seismic_amplitude
    assert len(builder._variables) == 6 
    v = next((v for v in builder._variables if v.name == "seismic_amplitude"), None)
    assert v is not None
    assert v.long_name == "Amplitude (dimensionless)"
    # NOTE: add_variable() stores dimensions as names
    assert set(v.dimensions) == {"inline", "crossline", "depth"}
    assert v.data_type == ScalarType.FLOAT64
    assert isinstance(v.compressor, Blosc)
    assert v.compressor.algorithm == "zstd"
    # NOTE: add_variable() stores coordinates as names
    assert set(v.coordinates) == {"cdp-x", "cdp-y"}
    assert v.metadata.stats_v1.count == 100 
    assert isinstance(v.metadata, VariableMetadata)
    assert v.metadata.units_v1.length == LengthUnitEnum.FOOT
    assert v.metadata.attributes["MGA"] == 51
    assert v.metadata.attributes["UnitSystem"] == "Imperial"  
    assert v.metadata.chunk_grid.name == "regular"
    assert v.metadata.chunk_grid.configuration.chunk_shape == [20]  
    assert v.metadata.stats_v1.count == 100  
    assert v.metadata.stats_v1.sum == 1215.1  
    assert v.metadata.stats_v1.sum_squares == 125.12  
    assert v.metadata.stats_v1.min == 5.61  
    assert v.metadata.stats_v1.max == 10.84  
    assert v.metadata.stats_v1.histogram.bin_centers == [1, 2]  
    assert v.metadata.stats_v1.histogram.counts == [10, 15]  

