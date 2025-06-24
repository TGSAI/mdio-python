
import pytest

from datetime import datetime

from mdio.schemas.chunk_grid import RectilinearChunkGrid, RectilinearChunkShape, RegularChunkGrid, RegularChunkShape
from mdio.schemas.dtype import ScalarType, StructuredType

from mdio.schemas.dimension import NamedDimension
from mdio.schemas.metadata import ChunkGridMetadata, UserAttributes
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder, contains_dimension, get_dimension, _BuilderState
from mdio.schemas.v1.stats import CenteredBinHistogram, Histogram, StatisticsMetadata, SummaryStatistics
from mdio.schemas.v1.units import AllUnits, LengthUnitEnum, LengthUnitModel

def test_add_dimension() -> None:
    """Test adding a dimension to the dataset builder."""
    builder = MDIODatasetBuilder("Test Dataset Builder")

    builder.add_dimension(name="inline", size=2, long_name="Inline dimension")
    assert len(builder._dimensions) == 1
    assert builder._dimensions[0] == NamedDimension(name="inline", size=2)  
    assert len(builder._variables) == 1
    assert builder._state == _BuilderState.HAS_DIMENSIONS


def test_add_dimension() -> None:
    """Test dimension builder state transitions and functionality."""
    builder = MDIODatasetBuilder("test_dataset")

    # First dimension should change state to HAS_DIMENSIONS and create a variable
    builder.add_dimension("x", 100, long_name="X Dimension")
    assert builder._state == _BuilderState.HAS_DIMENSIONS
    assert len(builder._dimensions) == 1  # noqa: PLR2004
    assert len(builder._variables) == 1  # noqa: PLR2004
    assert builder._dimensions[0].name == "x"
    assert builder._dimensions[0].size == 100  # noqa: PLR2004
    var0 = builder._variables[0]
    assert var0.name == "x"
    assert var0.long_name == "X Dimension"
    assert var0.data_type == ScalarType.INT32
    assert var0.dimensions[0].name == "x"

    # Adding another dimension should maintain state and create another variable
    builder.add_dimension("y", 200, data_type=ScalarType.UINT32)
    assert builder._state == _BuilderState.HAS_DIMENSIONS
    assert len(builder._dimensions) == 2  # noqa: PLR2004
    assert len(builder._variables) == 2  # noqa: PLR2004
    assert builder._dimensions[1].name == "y"
    assert builder._dimensions[1].size == 200  # noqa: PLR2004
    var1 = builder._variables[1]
    assert var1.name == "y"
    assert var1.data_type == ScalarType.UINT32
    assert var1.dimensions[0].name == "y"

    # TODO: Adding a dimension with the same: is allowed allowed (currently ignored) or should have raise an error?
    builder.add_dimension("x", 100, long_name="X Dimension")
    assert builder._state == _BuilderState.HAS_DIMENSIONS
    assert len(builder._dimensions) == 2  # noqa: PLR2004
    assert len(builder._variables) == 2  # noqa: PLR2004

    # Adding a dimension with the same name and different size throws an error
    with pytest.raises(ValueError, match="Dimension 'x' found but size 100 does not match expected size 200"):
        builder.add_dimension("x", 200, long_name="X Dimension") 
        assert builder._state == _BuilderState.HAS_DIMENSIONS  



def test_add_dimension_with_units() -> None:
    """Test adding dimensions with units."""
    builder = MDIODatasetBuilder("test_dataset")

    # Add dimension with units as a dictionary
    builder.add_dimension(
        "depth",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata={"unitsV1": {"length": "m"}},
    )
    assert len(builder._variables) == 1
    var0 = builder._variables[0]
    assert var0.name == "depth"
    assert var0.data_type == ScalarType.FLOAT32
    assert var0.metadata.units_v1.length == "m"

    # Add dimension with strongly-typed unit list of single-item
    builder.add_dimension(
        "length",
        size=100,
        data_type=ScalarType.FLOAT64,
        metadata=[AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT))],
    )
    assert len(builder._variables) == 2
    var1 = builder._variables[1]
    assert var1.name == "length"
    assert var1.data_type == ScalarType.FLOAT64
    assert var1.metadata.units_v1.length == "ft"


def test_add_dimension_with_attributes() -> None:
    """Test adding dimensions with attributes."""
    builder = MDIODatasetBuilder("test_dataset")

    # Add dimension with attributes as dictionary
    builder.add_dimension(
        "depth",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata={"attributes": {"MGA": 51, "UnitSystem": "Imperial"}}
    )
    assert len(builder._variables) == 1
    var0 = builder._variables[0]
    assert var0.name == "depth"
    assert var0.data_type == ScalarType.FLOAT32
    assert var0.metadata.attributes["MGA"] == 51  # noqa: PLR2004
    assert var0.metadata.attributes["UnitSystem"] == "Imperial"

    # Add dimension with strongly-typed attribute list
    builder.add_dimension(
        "length",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata=[UserAttributes(attributes={"MGA": 51, "UnitSystem": "Imperial"})]
    )
    assert len(builder._variables) == 2
    var1 = builder._variables[1]
    assert var1.name == "length"
    assert var1.data_type == ScalarType.FLOAT32
    assert var1.metadata.attributes["MGA"] == 51  # noqa: PLR2004
    assert var1.metadata.attributes["UnitSystem"] == "Imperial"


def test_add_dimension_with_chunk_grid() -> None:
    """Test adding dimensions with chunk grid."""
    builder = MDIODatasetBuilder("test_dataset")

    # Add dimension with chunk grid as dictionary
    builder.add_dimension(
        "depth",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata={"chunkGrid": {"name": "regular", "configuration": {"chunkShape": [20]}}},
    )
    assert len(builder._variables) == 1
    var0 = builder._variables[0]
    assert var0.name == "depth"
    assert var0.data_type == ScalarType.FLOAT32
    assert var0.metadata.chunk_grid.name == "regular"
    assert var0.metadata.chunk_grid.configuration.chunk_shape == [20]


    # Add dimension with strongly-typed chunk grid
    # TODO: It is not clear, how RectilinearChunkGrid can be mapped to a single dimension
    # grid_definition = RectilinearChunkGrid(configuration=RectilinearChunkShape(chunk_shape=[[2,3,4],[2,3,4]]))
    grid_definition = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=[20]))
    builder.add_dimension(
        "length",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata=[ChunkGridMetadata(chunk_grid=grid_definition)]
    )
    assert len(builder._variables) == 2
    var1 = builder._variables[1]
    assert var1.name == "length"
    assert var1.data_type == ScalarType.FLOAT32
    assert var1.metadata.chunk_grid.name == "regular"
    assert var1.metadata.chunk_grid.configuration.chunk_shape == [20]

def test_add_dimension_with_stats() -> None:
    """Test adding dimensions with stats."""
    builder = MDIODatasetBuilder("test_dataset")

    # TODO: Are multiple statistic object supported?
    # TODO: StatisticsMetadata accepts list[SummaryStatistics], what does this mean and does it need to be tested?

    # TODO:  What is the proper spelling 'statsV1' or 'stats_v1'? Needs to be documented.
    
    # Add dimension with strongly-typed stats
    builder.add_dimension(
        "depth",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata=[StatisticsMetadata(stats_v1=SummaryStatistics(
            count=100,
            sum=1215.1,
            sumSquares=125.12,
            min=5.61,
            max=10.84,
            # TODO: Also test EdgeDefinedHistogram
            histogram=CenteredBinHistogram(binCenters=[1, 2], counts=[10, 15])
        ))]
    )
    assert len(builder._variables) == 1
    var0 = builder._variables[0]
    assert var0.name == "depth"
    assert var0.data_type == ScalarType.FLOAT32
    assert var0.metadata.stats_v1.count == 100  # noqa: PLR2004
    assert var0.metadata.stats_v1.sum == 1215.1  # noqa: PLR2004

    # Add dimension with dictionary stats
    builder.add_dimension(
        "length",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata={
            "statsV1": {
                "count": 100,
                "sum": 1215.1,
                "sumSquares": 125.12,
                "min": 5.61,
                "max": 10.84,
                "histogram": {"binCenters": [1, 2], "counts": [10, 15]},
            }
        },
    )
    assert len(builder._variables) == 2
    var1 = builder._variables[1]
    assert var1.name == "length"
    assert var1.data_type == ScalarType.FLOAT32
    assert var1.metadata.stats_v1.count == 100  # noqa: PLR2004
    assert var1.metadata.stats_v1.sum == 1215.1  # noqa: PLR2004

def test_add_dimension_with_full_metadata() -> None:
    """Test adding dimensions with all metadata."""
    builder = MDIODatasetBuilder("test_dataset")

    # Add dimension with all metadata as dictionary
    builder.add_dimension(
        "depth",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata={
            "unitsV1": {"length": "m"},
            "attributes": {"MGA": 51},
            "chunkGrid": {"name": "regular", "configuration": {"chunkShape": [20]}},
            "statsV1": {
                "count": 100,
                "sum": 1215.1,
                "sumSquares": 125.12,
                "min": 5.61,
                "max": 10.84,
                "histogram": {"binCenters": [1, 2], "counts": [10, 15]},
            },
        },
    )

    assert len(builder._variables) == 1
    var0 = builder._variables[0]
    assert var0.name == "depth"
    assert var0.data_type == ScalarType.FLOAT32
    assert var0.metadata.units_v1.length == "m"
    assert var0.metadata.attributes["MGA"] == 51  # noqa: PLR2004
    assert var0.metadata.chunk_grid.name == "regular"
    assert var0.metadata.chunk_grid.configuration.chunk_shape == [20]  # noqa: PLR2004
    assert var0.metadata.stats_v1.count == 100  # noqa: PLR2004
    assert var0.metadata.stats_v1.sum == 1215.1  # noqa: PLR2004
    assert var0.metadata.stats_v1.sum_squares == 125.12  # noqa: PLR2004
    assert var0.metadata.stats_v1.min == 5.61  # noqa: PLR2004
    assert var0.metadata.stats_v1.max == 10.84  # noqa: PLR2004
    assert var0.metadata.stats_v1.histogram.bin_centers == [1, 2]  # noqa: PLR2004
    assert var0.metadata.stats_v1.histogram.counts == [10, 15]  # noqa: PLR2004

    # Add dimension with all strongly-typed metadata
    builder.add_dimension(
        "length",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata=[
            AllUnits(units_v1=LengthUnitModel(
                length=LengthUnitEnum.FOOT)),
            UserAttributes(
                attributes={"MGA": 51, "UnitSystem": "Imperial"}),
            ChunkGridMetadata(
                chunk_grid=RegularChunkGrid(
                    configuration=RegularChunkShape(
                        chunk_shape=[20]))),
            StatisticsMetadata(stats_v1=SummaryStatistics(
                count=100,
                sum=1215.1,
                sumSquares=125.12,
                min=5.61,
                max=10.84,
                histogram=CenteredBinHistogram(
                    binCenters=[1, 2], 
                    counts=[10, 15])))
        ]
    )

    assert len(builder._variables) == 2
    var1 = builder._variables[1]
    assert var1.name == "length"
    assert var1.data_type == ScalarType.FLOAT32
    assert var1.metadata.units_v1.length == "ft"
    assert var1.metadata.attributes["MGA"] == 51  # noqa: PLR2004
    assert var1.metadata.attributes["UnitSystem"] == "Imperial"  # noqa: PLR2004
    assert var1.metadata.chunk_grid.name == "regular"
    assert var1.metadata.chunk_grid.configuration.chunk_shape == [20]  # noqa: PLR2004
    assert var1.metadata.stats_v1.count == 100  # noqa: PLR2004
    assert var1.metadata.stats_v1.sum == 1215.1  # noqa: PLR2004
    assert var1.metadata.stats_v1.sum_squares == 125.12  # noqa: PLR2004
    assert var1.metadata.stats_v1.min == 5.61  # noqa: PLR2004
    assert var1.metadata.stats_v1.max == 10.84  # noqa: PLR2004
    assert var1.metadata.stats_v1.histogram.bin_centers == [1, 2]  # noqa: PLR2004
    assert var1.metadata.stats_v1.histogram.counts == [10, 15]  # noqa: PLR2004


    # j = builder.build().json()
    # print(j)