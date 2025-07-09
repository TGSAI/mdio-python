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

from .helpers import validate_builder
from .helpers import validate_variable


def test_add_variable_no_coords() -> None:
    """Test adding variable. Check the state transition and validate required parameters.."""
    builder = MDIODatasetBuilder("test_dataset")
    validate_builder(builder, _BuilderState.INITIAL, n_dims=0, n_coords=0, n_var=0)

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
    validate_builder(builder, _BuilderState.HAS_VARIABLES, n_dims=3, n_coords=0, n_var=1)
    validate_variable(builder, "amplitude",
                       dims=[("inline", 100), ("crossline", 200), ("depth", 300)],
                       coords=None,
                       dtype=ScalarType.FLOAT32)

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
    validate_builder(builder, _BuilderState.HAS_VARIABLES, n_dims=3, n_coords=2, n_var=3)
    validate_variable(builder, "ampl",
                       dims=[("inline", 100), ("crossline", 200), ("depth", 300)],
                       coords=["inline", "crossline"],
                       dtype=ScalarType.FLOAT32)

    # Add non-dim coordinates (e.g., 2D coordinates)
    builder.add_coordinate(
        "cdp-x", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT32)
    builder.add_coordinate(
        "cdp-y", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT32)

    # Add a variable with pre-defined dimension and non-dimension coordinates
    builder.add_variable("ampl2",
                         dimensions=["inline", "crossline", "depth"],
                         coordinates=["inline", "crossline", "cdp-x", "cdp-y"],
                         data_type=ScalarType.FLOAT32)
    validate_builder(builder, _BuilderState.HAS_VARIABLES, n_dims=3, n_coords=4, n_var=6)
    validate_variable(builder, "ampl2",
                       dims=[("inline", 100), ("crossline", 200), ("depth", 300)],
                       coords=["inline", "crossline", "cdp-x", "cdp-y"],
                       dtype=ScalarType.FLOAT32)


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
    validate_builder(builder, _BuilderState.HAS_VARIABLES, n_dims=3, n_coords=3, n_var=4)
    v = validate_variable(builder, "ampl",
                                   dims=[("inline", 100), ("crossline", 200), ("depth", 300)],
                                   coords=None,
                                   dtype=ScalarType.FLOAT32)
    assert v.long_name is None  # Default value
    assert v.compressor is None  # Default value
    assert v.coordinates is None  # Default value
    assert v.metadata is None  # Default value


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
    builder.add_coordinate(
        "cdp-x", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT64)
    builder.add_coordinate(
        "cdp-y", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT64)

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
    validate_builder(builder, _BuilderState.HAS_VARIABLES, n_dims=3, n_coords=5, n_var=6)
    v = validate_variable(builder, "ampl",
                           dims=[("inline", 100), ("crossline", 200), ("depth", 300)],
                           coords=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
                           dtype=ScalarType.FLOAT32)
    assert v.long_name == "Amplitude (dimensionless)"
    assert isinstance(v.compressor, Blosc)
    assert v.compressor.algorithm == "zstd"
    assert len(v.coordinates) == 5
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
