# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 dataset_builder.add_dimension() public API."""

import pytest

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.chunk_grid import RegularChunkShape
from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.dataset_builder import _BuilderState
from mdio.schemas.v1.dataset_builder import _get_dimension
from mdio.schemas.v1.stats import CenteredBinHistogram
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.stats import SummaryStatistics
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel


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
    assert builder._state == _BuilderState.HAS_DIMENSIONS
    assert len(builder._dimensions) == 1
    assert _get_dimension(builder._dimensions, "x", 100) is not None

    # Validate that we have created a dimension variable and 
    # that variable has the embedded NamedDimension
    assert len(builder._variables) == 1  
    var_x = next(e for e in builder._variables if e.name == "x")
    assert var_x is not None
    assert len(var_x.dimensions) == 1
    # Validate that the dimension variable has the NamedDimension
    assert _get_dimension(var_x.dimensions, "x", 100) is not None
    assert var_x.long_name == "'x' dimension variable"
    assert var_x.data_type == ScalarType.INT32
    assert var_x.compressor is None
    assert var_x.coordinates is None
    assert var_x.metadata is None

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

def test_add_dimension_with_defaults() -> None:
    """Test dimension builder state transitions and functionality."""
    builder = MDIODatasetBuilder("test_dataset")

    # First dimension should change state to HAS_DIMENSIONS and create a variable
    builder.add_dimension("x", 100)
    assert builder._state == _BuilderState.HAS_DIMENSIONS
    assert len(builder._dimensions) == 1  
    # Validate that the dimension builder has the NamedDimension
    assert _get_dimension(builder._dimensions, "x", 100) is not None
    var_x = next((e for e in builder._variables if e.name == "x"), None)
    assert var_x is not None
    assert var_x.name == "x"
    assert var_x.long_name == "'x' dimension variable"
    assert var_x.data_type == ScalarType.INT32
    assert var_x.compressor is None
    assert var_x.coordinates is None
    assert var_x.metadata is None

def test_add_dimension_with_units() -> None:
    """Test adding dimensions with units."""
    builder = MDIODatasetBuilder("test_dataset")

    # Add dimension with strongly-typed unit list of single-item
    builder.add_dimension(
        "length",
        size=100,
        var_data_type=ScalarType.FLOAT64,
        var_metadata_info=[AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT))]
    )
    assert len(builder._variables) == 1
    var0 = builder._variables[0]
    assert var0.name == "length"
    assert var0.long_name == "'length' dimension variable"
    assert var0.data_type == ScalarType.FLOAT64
    assert var0.compressor is None
    assert var0.coordinates is None
    assert var0.metadata.units_v1.length == LengthUnitEnum.FOOT

def test_add_dimension_with_attributes() -> None:
    """Test adding dimensions with attributes."""
    builder = MDIODatasetBuilder("test_dataset")

    builder.add_dimension(
        "length",
        size=100,
        var_data_type=ScalarType.FLOAT32,
        var_metadata_info=[UserAttributes(attributes={"MGA": 51, "UnitSystem": "Imperial"})],
    )
    assert len(builder._variables) == 1
    var0 = builder._variables[0]
    assert var0.name == "length"
    assert var0.data_type == ScalarType.FLOAT32
    assert var0.metadata.attributes["MGA"] == 51
    assert var0.metadata.attributes["UnitSystem"] == "Imperial"


def test_add_dimension_with_chunk_grid() -> None:
    """Test adding dimensions with chunk grid."""
    builder = MDIODatasetBuilder("test_dataset")

    grid_definition = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=[20]))
    builder.add_dimension(
        "length",
        size=100,
        var_data_type=ScalarType.FLOAT32,
        var_metadata_info=[ChunkGridMetadata(chunk_grid=grid_definition)],
    )
    assert len(builder._variables) == 1
    var0 = builder._variables[0]
    assert var0.name == "length"
    assert var0.data_type == ScalarType.FLOAT32
    assert var0.metadata.chunk_grid.name == "regular"
    assert var0.metadata.chunk_grid.configuration.chunk_shape == [20]


def test_add_dimension_with_stats() -> None:
    """Test adding dimensions with stats."""
    builder = MDIODatasetBuilder("test_dataset")

    builder.add_dimension(
        "depth",
        size=100,
        var_data_type=ScalarType.FLOAT32,
        var_metadata_info=[
            StatisticsMetadata(
                stats_v1=SummaryStatistics(
                    count=100,
                    sum=1215.1,
                    sumSquares=125.12,
                    min=5.61,
                    max=10.84,
                    # TODO(DmitriyRepin, #0): Also test EdgeDefinedHistogram
                    histogram=CenteredBinHistogram(binCenters=[1, 2], counts=[10, 15]),
                )
            )
        ],
    )
    assert len(builder._variables) == 1
    var0 = builder._variables[0]
    assert var0.name == "depth"
    assert var0.data_type == ScalarType.FLOAT32
    assert var0.metadata.stats_v1.count == 100  
    assert var0.metadata.stats_v1.sum == 1215.1  


def test_add_dimension_with_full_metadata() -> None:
    """Test adding dimensions with all metadata."""
    builder = MDIODatasetBuilder("test_dataset")

    builder.add_dimension(
        "length",
        size=100,
        var_data_type=ScalarType.FLOAT32,
        var_metadata_info=[
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
        ],
    )

    assert len(builder._variables) == 1
    var0 = builder._variables[0]
    assert var0.name == "length"
    assert var0.data_type == ScalarType.FLOAT32
    assert var0.metadata.units_v1.length == LengthUnitEnum.FOOT
    assert var0.metadata.attributes["MGA"] == 51
    assert var0.metadata.attributes["UnitSystem"] == "Imperial"
    assert var0.metadata.chunk_grid.name == "regular"
    assert var0.metadata.chunk_grid.configuration.chunk_shape == [20]  
    assert var0.metadata.stats_v1.count == 100  
    assert var0.metadata.stats_v1.sum == 1215.1  
    assert var0.metadata.stats_v1.sum_squares == 125.12  
    assert var0.metadata.stats_v1.min == 5.61  
    assert var0.metadata.stats_v1.max == 10.84  
    assert var0.metadata.stats_v1.histogram.bin_centers == [1, 2]  
    assert var0.metadata.stats_v1.histogram.counts == [10, 15]  
