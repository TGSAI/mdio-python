# ruff: noqa: PLR2004 
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert. 
# Thus, disable it for this file
"""Tests the schema v1 dataset_builder internal methods."""

from datetime import UTC
from datetime import datetime

import pytest
from pydantic import Field

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.chunk_grid import RegularChunkShape
from mdio.schemas.core import StrictModel
from mdio.schemas.dimension import NamedDimension
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.dataset_builder import _to_dictionary
from mdio.schemas.v1.dataset_builder import contains_dimension
from mdio.schemas.v1.dataset_builder import get_dimension
from mdio.schemas.v1.dataset_builder import get_dimension_names
from mdio.schemas.v1.stats import CenteredBinHistogram
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.stats import SummaryStatistics
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.variable import AllUnits
from mdio.schemas.v1.variable import CoordinateMetadata
from mdio.schemas.v1.variable import UserAttributes
from mdio.schemas.v1.variable import VariableMetadata


def test__get_dimension_by_name() -> None:
    """Test getting a dimension by name from the list of dimensions."""
    dimensions = [NamedDimension(name="inline", size=2), NamedDimension(name="crossline", size=3)]

    assert get_dimension([], "inline") is None
    assert get_dimension(dimensions, "inline") == NamedDimension(name="inline", size=2)
    assert get_dimension(dimensions, "crossline") == NamedDimension(name="crossline", size=3)
    assert get_dimension(dimensions, "time") is None

    with pytest.raises(TypeError, match="Expected str, got NoneType"):
        get_dimension(dimensions, None)
    with pytest.raises(TypeError, match="Expected str, got int"):
        get_dimension(dimensions, 42)
    with pytest.raises(
        ValueError, match="Dimension 'inline' found but size 2 does not match expected size 200"
    ):
        get_dimension(dimensions, "inline", size=200)


def test__contains_dimension() -> None:
    """Test if a dimension with a given name exists in the list of dimensions."""
    dimensions = [NamedDimension(name="inline", size=2), NamedDimension(name="crossline", size=3)]

    assert contains_dimension([], "inline") is False

    assert contains_dimension(dimensions, "inline") is True
    assert contains_dimension(dimensions, "crossline") is True
    assert contains_dimension(dimensions, "time") is False

    with pytest.raises(TypeError, match="Expected str or NamedDimension, got NoneType"):
        contains_dimension(dimensions, None)
    with pytest.raises(TypeError, match="Expected str or NamedDimension, got int"):
        contains_dimension(dimensions, 42)
    with pytest.raises(
        ValueError, match="Dimension 'inline' found but size 2 does not match expected size 200"
    ):
        contains_dimension(dimensions, NamedDimension(name="inline", size=200))


def test_get_dimension_names() -> None:
    """Test getting a list of dimension names from list[NamedDimension | str]."""
    empty_list = get_dimension_names(None)
    assert empty_list is not None
    assert isinstance(empty_list, list)
    assert len(empty_list) == 0

    empty_list = get_dimension_names([])
    assert empty_list is not None
    assert isinstance(empty_list, list)
    assert len(empty_list) == 0

    dim_list = get_dimension_names([
        NamedDimension(name="inline", size=2), 
        "amplitude",
        NamedDimension(name="crossline", size=3)
    ])
    assert dim_list is not None
    assert isinstance(dim_list, list)
    assert set(dim_list) == {"inline", "amplitude", "crossline"}


def test_add_dimensions_if_needed() -> None:
    """Test adding named dimensions to a dataset."""
    builder = MDIODatasetBuilder("Test Dataset Builder")
    #
    # Validate initial state
    #
    assert builder._dimensions is not None
    assert len(builder._dimensions) == 0

    #
    # Validate that adding empty dimensions does not change the state
    #
    added_dims = builder._add_dimensions_if_needed(None)
    assert len(builder._dimensions) == 0
    assert len(added_dims) == 0
    added_dims = builder._add_dimensions_if_needed([])
    assert len(builder._dimensions) == 0
    assert len(added_dims) == 0
    added_dims = builder._add_dimensions_if_needed({})
    assert len(builder._dimensions) == 0
    assert len(added_dims) == 0

    #
    # Add named dimensions
    #
    inline_dim = NamedDimension(name="inline", size=2)
    added_dims = builder._add_dimensions_if_needed([inline_dim])
    assert len(builder._dimensions) == 1
    assert len(added_dims) == 1
    assert contains_dimension(added_dims, inline_dim)

    crossline_dim = NamedDimension(name="crossline", size=3)
    time_dim = NamedDimension(name="time", size=4)
    added_dims = builder._add_dimensions_if_needed([crossline_dim, time_dim])
    assert len(builder._dimensions) == 3
    assert contains_dimension(builder._dimensions, inline_dim)
    assert contains_dimension(builder._dimensions, crossline_dim)
    assert contains_dimension(builder._dimensions, time_dim)
    assert contains_dimension(added_dims, crossline_dim)
    assert contains_dimension(added_dims, time_dim)

    #
    # Add invalid object type
    #
    with pytest.raises(TypeError, match="Expected NamedDimension or str, got int"):
        builder._add_dimensions_if_needed([42])
    # Assert that the number of dimensions has not increased
    assert len(builder._dimensions) == 3

def test__add_dimensions_if_one_already_exists() -> None:
    """Test adding existing named dimensions to a dataset."""
    builder = MDIODatasetBuilder("Test Dataset Builder")

    inline_dim = NamedDimension(name="inline", size=2)
    crossline_dim = NamedDimension(name="crossline", size=3)
    time_dim = NamedDimension(name="time", size=4)
    #
    # Add dimensions with the same names again does nothing
    # (make sure we are passing different instances)
    #
    inline_dim2 = NamedDimension(name=inline_dim.name, size=inline_dim.size)
    crossline_dim2 = NamedDimension(name=crossline_dim.name, size=crossline_dim.size)
    time_dim2 = NamedDimension(name=time_dim.name, size=time_dim.size)
    builder._add_dimensions_if_needed([inline_dim2, crossline_dim2, time_dim2])
    added_dims = builder._add_dimensions_if_needed([inline_dim2, crossline_dim2, time_dim2])
    # Validate that the dimensions and variables are not duplicated
    assert len(builder._dimensions) == 3
    assert contains_dimension(builder._dimensions, inline_dim)
    assert contains_dimension(builder._dimensions, crossline_dim)
    assert contains_dimension(builder._dimensions, time_dim)
    assert len(added_dims) == 0

    # Add dimensions with the same name, but different size again
    with pytest.raises(
        ValueError, match="Dimension 'inline' found but size 2 does not match expected size 200"
    ):
        builder._add_dimensions_if_needed([NamedDimension(name="inline", size=200)])
    # Assert that the number of dimensions has not increased
    assert len(builder._dimensions) == 3

    #
    # Add existing dimension using its name
    #
    added_dims = builder._add_dimensions_if_needed(["inline", "crossline"])
    assert len(builder._dimensions) == 3
    assert contains_dimension(builder._dimensions, inline_dim)
    assert contains_dimension(builder._dimensions, crossline_dim)
    assert contains_dimension(builder._dimensions, time_dim)
    assert len(added_dims) == 0

    #
    # Add non-existing dimension using its name is not allowed
    #
    with pytest.raises(ValueError, match="Pre-existing dimension named 'offset' is not found"):
        builder._add_dimensions_if_needed(["offset"])
    assert len(builder._dimensions) == 3
    assert contains_dimension(builder._dimensions, inline_dim)
    assert contains_dimension(builder._dimensions, crossline_dim)
    assert contains_dimension(builder._dimensions, time_dim)

def test__to_dictionary() -> None:
    """Test converting a BaseModel to a dictionary."""
    with pytest.raises(TypeError, match="Expected BaseModel, dict or list, got datetime"):
        # This should raise an error because datetime is not a BaseModel
        _to_dictionary(datetime.now(UTC))

    class SomeModel(StrictModel):
        count: int = Field(default=None, description="Samples count")
        samples: list[float] = Field(default_factory=list, description="Samples.")
        created: datetime = Field(
            default_factory=datetime.now, description="Creation time with TZ info."
        )

    m = SomeModel(count=3, 
                  samples=[1.0, 2.0, 3.0], 
                  created=datetime(2023, 10, 1, 12, 0, 0, tzinfo=UTC))
    result = _to_dictionary(m)
    assert isinstance(result, dict)
    assert result == {"count": 3, "created": "2023-10-01T12:00:00Z", "samples": [1.0, 2.0, 3.0]}

    d = {
        "count": 3,
        "samples": [1.0, 2.0, 3.0],
        "created": datetime(2023, 10, 1, 12, 0, 0, tzinfo=UTC)}
    result = _to_dictionary(m)
    assert isinstance(result, dict)
    assert result == {"count": 3, "created": "2023-10-01T12:00:00Z", "samples": [1.0, 2.0, 3.0]}


# def test__make_coordinate_metadata() -> None:
#     """Test creating VariableMetadata from a strongly-typed list of AllUnits or UserAttributes."""
#     units = AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT))
#     attrs = UserAttributes(attributes={"MGA": 51, "UnitSystem": "Imperial"})
#     meta_list = [units, attrs]

#     # Assume that multiple attributes are allowed
#     metadata = _make_coordinate_metadata(meta_list)
#     assert isinstance(metadata, CoordinateMetadata)
#     assert metadata.units_v1.length == "ft"
#     assert metadata.attributes["MGA"] == 51
#     assert metadata.attributes["UnitSystem"] == "Imperial"

#     meta_list = ["ft"]
#     with pytest.raises(TypeError, match="Expected BaseModel, dict or list, got str"):
#         _make_variable_metadata(meta_list)

# def test__make_variable_metadata() -> None:
#     """Test creating VariableMetadata from a strongly-typed list of AllUnits or UserAttributes."""
#     units = AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT))
#     attrs = UserAttributes(attributes={"MGA": 51, "UnitSystem": "Imperial"})
#     chgrd = ChunkGridMetadata(
#                 chunk_grid=RegularChunkGrid(
#                     configuration=RegularChunkShape(chunk_shape=[20])))
#     stats = StatisticsMetadata(
#                 stats_v1=SummaryStatistics(
#                     count=100,
#                     sum=1215.1,
#                     sumSquares=125.12,
#                     min=5.61,
#                     max=10.84,
#                     histogram=CenteredBinHistogram(binCenters=[1, 2], counts=[10, 15])))
#     metadata_info = [units, attrs, chgrd, stats]
#     metadata = _make_variable_metadata(metadata_info)
#     assert isinstance(metadata, VariableMetadata)
#     assert metadata.units_v1.length == "ft"
#     assert metadata.attributes["MGA"] == 51  
#     assert metadata.attributes["UnitSystem"] == "Imperial"  
#     assert metadata.chunk_grid.name == "regular"
#     assert metadata.chunk_grid.configuration.chunk_shape == [20]  
#     assert metadata.stats_v1.count == 100  
#     assert metadata.stats_v1.sum == 1215.1  
#     assert metadata.stats_v1.sum_squares == 125.12  
#     assert metadata.stats_v1.min == 5.61  
#     assert metadata.stats_v1.max == 10.84  
#     assert metadata.stats_v1.histogram.bin_centers == [1, 2]  
#     assert metadata.stats_v1.histogram.counts == [10, 15]  

#     meta_list = ["ft"]
#     with pytest.raises(TypeError, match="Expected BaseModel, dict or list, got str"):
#         _make_variable_metadata(meta_list)
