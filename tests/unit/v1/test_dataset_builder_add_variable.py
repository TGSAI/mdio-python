# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 Variable public API."""

import pytest

from mdio.schemas import builder
from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.chunk_grid import RegularChunkShape
from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder, _get_named_dimension
from mdio.schemas.v1.dataset_builder import _BuilderState
from mdio.schemas.v1.stats import CenteredBinHistogram
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.stats import SummaryStatistics
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.variable import VariableMetadata


def test_add_variable_no_coords() -> None:
    """Test adding variable. Check the state transition and validate required parameters.."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder._state == _BuilderState.INITIAL

    # Validate: Must add at least one dimension before adding variables
    msg = "Must add at least one dimension before adding variables"
    with pytest.raises(ValueError, match=msg):
        builder.add_variable("amplitude", dimensions=[
                             "speed"], data_type=ScalarType.FLOAT32)

   # Add dimension before we can add a data variable
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 200)
    builder.add_dimension("depth", 300)

    # Validate: required parameters must be preset
    bad_name = None
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_variable(bad_name, dimensions=[
                             "speed"], data_type=ScalarType.FLOAT32)
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_variable(
            "", dimensions=["speed"], data_type=ScalarType.FLOAT32)
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_variable("bad_amplitude", dimensions=None,
                             data_type=ScalarType.FLOAT32)
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_variable("bad_amplitude", dimensions=[],
                             data_type=ScalarType.FLOAT32)

    # Validate: Add a variable using non-existent dimensions is not allowed
    msg = "Pre-existing dimension named 'il' is not found"
    with pytest.raises(ValueError, match=msg):
        builder.add_variable("bad_amplitude",
                             dimensions=["il", "xl", "depth"],
                             data_type=ScalarType.FLOAT32)

    # Add a variable without coordinates
    builder.add_variable("amplitude",
                         dimensions=["inline", "crossline", "depth"],
                         data_type=ScalarType.FLOAT32)
    assert builder._state == _BuilderState.HAS_VARIABLES
    assert len(builder._dimensions) == 3
    assert len(builder._variables) == 1
    assert len(builder._coordinates) == 0

    # Validate the structure of the created variable
    var_ampl = next((e for e in builder._variables if e.name == "amplitude"), None)
    assert var_ampl is not None
    # Validate that dimensions are stored as NamedDimensions
    assert _get_named_dimension(var_ampl.dimensions, "inline", 100) is not None
    assert _get_named_dimension(var_ampl.dimensions, "crossline", 200) is not None
    assert _get_named_dimension(var_ampl.dimensions, "depth", 300) is not None
    # Validate that no coordinates are set
    assert var_ampl.coordinates is None

    # Validate: adding a variable with the same name twice is not allowed
    msg = "Adding variable with the same name twice is not allowed"
    with pytest.raises(ValueError, match=msg):
        builder.add_variable("amplitude",
                             dimensions=["inline", "crossline", "depth"],
                             data_type=ScalarType.FLOAT32)


def test_add_variable_with_coords() -> None:
    """Test adding variable. Check the state transition and validate required parameters.."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 200)
    builder.add_dimension("depth", 300)

    # Add dimension coordinates before we can add a data variable
    builder.add_coordinate("inline", dimensions=["inline"], data_type=ScalarType.UINT32)
    builder.add_coordinate("crossline", dimensions=["crossline"], data_type=ScalarType.UINT32)

    # Validate: adding a variable with a coordinate that has not been pre-created is not allowed
    msg = "Pre-existing coordinate named 'depth' is not found"
    with pytest.raises(ValueError, match=msg):
        builder.add_variable("ampl",
                             dimensions=["inline", "crossline", "depth"],
                             coordinates=["inline", "crossline", "depth"],
                             data_type=ScalarType.FLOAT32)

    # Add a variable with pre-defined dimension coordinates
    builder.add_variable("ampl",
                         dimensions=["inline", "crossline", "depth"],
                         coordinates=["inline", "crossline"],
                         data_type=ScalarType.FLOAT32)

    assert builder._state == _BuilderState.HAS_VARIABLES
    assert len(builder._dimensions) == 3
    # 2 dim coordinate variables + 1 data variables
    assert len(builder._variables) == 3
    assert len(builder._coordinates) == 2

    # Validate: the structure of the created variable
    var_ampl = next((e for e in builder._variables if e.name == "ampl"), None)
    assert var_ampl is not None
    # Validate: that dimensions are stored as NamedDimensions
    assert len(var_ampl.dimensions) == 3
    assert _get_named_dimension(var_ampl.dimensions, "inline", 100) is not None
    assert _get_named_dimension(var_ampl.dimensions, "crossline", 200) is not None
    assert _get_named_dimension(var_ampl.dimensions, "depth", 300) is not None
    assert len(var_ampl.coordinates) == 2
    # Validate that dim coordinates "inline" and "crossline" are set
    assert builder._get_coordinate(var_ampl.coordinates, "inline") is not None
    assert builder._get_coordinate(var_ampl.coordinates, "crossline") is not None
    # "depth" coordinate is not set

    # Add non-dim coordinates (e.g., 2D coordinates)
    builder.add_coordinate("cdp-x", dimensions=["inline", "crossline"])
    builder.add_coordinate("cdp-y", dimensions=["inline", "crossline"])

    # Add a variable with pre-defined dimension and non-dimension coordinates
    builder.add_variable("ampl2",
                         dimensions=["inline", "crossline", "depth"],
                         coordinates=["inline", "crossline", "cdp-x", "cdp-y"],
                         data_type=ScalarType.FLOAT32)

    assert builder._state == _BuilderState.HAS_VARIABLES
    assert len(builder._dimensions) == 3
    # 2 dim coordinate variables + 2 non-dim coordinate variables + 1 data variables
    assert len(builder._variables) == 6
    assert len(builder._coordinates) == 4

    # Validate: the structure of the created variable
    var_ampl2 = next((e for e in builder._variables if e.name == "ampl2"), None)
    assert var_ampl2 is not None
    # Validate: that dimensions are stored as NamedDimensions
    assert len(var_ampl2.dimensions) == 3
    assert _get_named_dimension(var_ampl2.dimensions, "inline", 100) is not None
    assert _get_named_dimension(var_ampl2.dimensions, "crossline", 200) is not None
    assert _get_named_dimension(var_ampl2.dimensions, "depth", 300) is not None
    assert len(var_ampl2.coordinates) == 4
    # Validate that dim coordinates "inline" and "crossline" are set
    assert builder._get_coordinate(var_ampl2.coordinates, "inline") is not None
    assert builder._get_coordinate(var_ampl2.coordinates, "crossline") is not None
    # "depth" coordinate is not set
    # Validate that non-dimension coordinates "cdp-x" and "cdp-y"
    assert builder._get_coordinate(var_ampl2.coordinates, "cdp-x") is not None
    assert builder._get_coordinate(var_ampl2.coordinates, "cdp-y") is not None


def test_add_variable_with_defaults() -> None:
    """Test adding variable with default arguments."""
    builder = MDIODatasetBuilder("test_dataset")
    # Add dimensions before we can add a data variables
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 200)
    builder.add_dimension("depth", 300)
    # Add dimension coordinates
    builder.add_coordinate("inline", dimensions=["inline"], data_type=ScalarType.UINT32)
    builder.add_coordinate("crossline", dimensions=["crossline"], data_type=ScalarType.UINT32)
    builder.add_coordinate("depth", dimensions=["depth"], data_type=ScalarType.UINT32,
                           metadata_info=[
                               AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))
                           ]) 

    # Add data variable using defaults
    builder.add_variable("ampl",
                         dimensions=["inline", "crossline", "depth"],
                         data_type=ScalarType.FLOAT32)
    assert len(builder._dimensions) == 3
    # 3 dim coordinate variables + 1 data variable = 4
    assert len(builder._variables) == 4
    assert len(builder._coordinates) == 3

    # Validate: the structure of the created variable
    var_ampl = next((e for e in builder._variables if e.name == "ampl"), None)
    assert var_ampl is not None
    assert var_ampl.name == "ampl"
    assert var_ampl.long_name is None  # Default value
    # Validate: that dimensions are stored as NamedDimensions
    assert len(var_ampl.dimensions) == 3
    assert _get_named_dimension(var_ampl.dimensions, "inline", 100) is not None
    assert _get_named_dimension(var_ampl.dimensions, "crossline", 200) is not None
    assert _get_named_dimension(var_ampl.dimensions, "depth", 300) is not None
    assert var_ampl.data_type == ScalarType.FLOAT32
    assert var_ampl.compressor is None  # Default value
    assert var_ampl.coordinates is None  # Default value
    # Validate: the variable has the expected properties
    assert var_ampl.metadata is None  # Default value


def test_add_variable_full_parameters() -> None:
    """Test adding variable with full parameters."""
    builder = MDIODatasetBuilder("test_dataset")
    # Add dimensions before we can add a data variables
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 200)
    builder.add_dimension("depth", 300)

    # Add dimension coordinates
    builder.add_coordinate("inline", dimensions=["inline"], data_type=ScalarType.UINT32)
    builder.add_coordinate("crossline", dimensions=["crossline"], data_type=ScalarType.UINT32)
    builder.add_coordinate("depth", dimensions=["depth"], data_type=ScalarType.UINT32)

    # Add coordinates before we can add a data variable
    builder.add_coordinate("cdp-x", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT64)
    builder.add_coordinate("cdp-y", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT64)

    # Add data variable with full parameters
    builder.add_variable("ampl",
                         long_name="Amplitude (dimensionless)",
                         dimensions=["inline", "crossline", "depth"],
                         data_type=ScalarType.FLOAT32,
                         compressor=Blosc(algorithm="zstd"),
                         coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
                         metadata_info=[
                             AllUnits(units_v1=LengthUnitModel(
                                 length=LengthUnitEnum.FOOT)),
                             UserAttributes(
                                 attributes={"MGA": 51, "UnitSystem": "Imperial"}),
                             ChunkGridMetadata(
                                 chunk_grid=RegularChunkGrid(
                                     configuration=RegularChunkShape(chunk_shape=[20]))
                             ),
                             StatisticsMetadata(
                                 stats_v1=SummaryStatistics(
                                     count=100,
                                     sum=1215.1,
                                     sumSquares=125.12,
                                     min=5.61,
                                     max=10.84,
                                     histogram=CenteredBinHistogram(
                                         binCenters=[1, 2], counts=[10, 15]),
                                 )
                             ),
                         ])
    # Validate: the state of the builder
    assert builder._state == _BuilderState.HAS_VARIABLES
    assert len(builder._dimensions) == 3
    # 3 dim coords + 2 non-dim coords = 5
    assert len(builder._coordinates) == 5
    # 3 dim coord + 2 non-dim coords, and 1 data variable
    assert len(builder._variables) == 6

     # Validate: the structure of the created variable
    v = next((v for v in builder._variables if v.name == "ampl"), None)
    assert v is not None
    assert v.name == "ampl"
    assert v.long_name == "Amplitude (dimensionless)"
    assert len(v.dimensions) == 3
    assert _get_named_dimension(v.dimensions, "inline", 100) is not None
    assert _get_named_dimension(v.dimensions, "crossline", 200) is not None
    assert _get_named_dimension(v.dimensions, "depth", 300) is not None
    assert v.data_type == ScalarType.FLOAT32
    assert isinstance(v.compressor, Blosc)
    assert v.compressor.algorithm == "zstd"
    assert len(v.coordinates) == 5
    assert builder._get_coordinate(v.coordinates, "inline") is not None
    assert builder._get_coordinate(v.coordinates, "crossline") is not None
    assert builder._get_coordinate(v.coordinates, "depth") is not None
    assert builder._get_coordinate(v.coordinates, "cdp-x") is not None
    assert builder._get_coordinate(v.coordinates, "cdp-y") is not None
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
