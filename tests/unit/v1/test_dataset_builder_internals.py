
from datetime import datetime
from pydantic import BaseModel, Field
import pytest
from mdio.schemas.core import StrictModel
from mdio.schemas.dimension import NamedDimension
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder, contains_dimension, get_dimension, to_dictionary
from mdio.schemas.v1.units import LengthUnitEnum, LengthUnitModel
from mdio.schemas.v1.variable import VariableMetadata, AllUnits, UserAttributes

def test__get_dimension() -> None:
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
    with pytest.raises(ValueError, match="Dimension 'inline' found but size 2 does not match expected size 200"):
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
    with pytest.raises(ValueError, match="Dimension 'inline' found but size 2 does not match expected size 200"):
        contains_dimension(dimensions, NamedDimension(name="inline", size=200))   



def test__add_named_dimensions() -> None:
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
    added_dims = builder._add_named_dimensions(None)
    assert len(builder._dimensions) == 0
    assert len(added_dims) == 0
    added_dims = builder._add_named_dimensions([])
    assert len(builder._dimensions) == 0
    assert len(added_dims) == 0
    added_dims = builder._add_named_dimensions({})
    assert len(builder._dimensions) == 0
    assert len(added_dims) == 0

    #
    # Add named dimensions
    #
    inline_dim = NamedDimension(name="inline", size=2)
    added_dims = builder._add_named_dimensions([inline_dim])
    assert len(builder._dimensions) == 1
    assert len(added_dims) == 1
    assert contains_dimension(added_dims, inline_dim)

    crossline_dim = NamedDimension(name="crossline", size=3)
    time_dim = NamedDimension(name="time", size=4)
    added_dims = builder._add_named_dimensions([crossline_dim, time_dim])
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
        builder._add_named_dimensions([42])
        assert len(builder._dimensions) == 3

    #
    # Add dimensions with the same names again does nothing
    # (make sure we are passing different instances)
    #
    inline_dim2 = NamedDimension(name=inline_dim.name, size=inline_dim.size)
    crossline_dim2 = NamedDimension(name=crossline_dim.name, size=crossline_dim.size)
    time_dim2 = NamedDimension(name=time_dim.name, size=time_dim.size)
    added_dims = builder._add_named_dimensions([inline_dim2, crossline_dim2, time_dim2])
    # Validate that the dimensions and variables are not duplicated
    assert len(builder._dimensions) == 3
    assert contains_dimension(builder._dimensions, inline_dim)
    assert contains_dimension(builder._dimensions, crossline_dim)
    assert contains_dimension(builder._dimensions, time_dim)
    assert len(added_dims) == 0

    # Add dimensions with the same name, but different size again
    with pytest.raises(ValueError, match="Dimension 'inline' found but size 2 does not match expected size 200"):
        inline_dim2 = NamedDimension(name="inline", size=200)
        builder._add_named_dimensions([inline_dim2])
        assert len(builder._dimensions) == 3
    #
    # Add existing dimension using its name
    #
    added_dims = builder._add_named_dimensions(["inline", "crossline"])
    assert len(builder._dimensions) == 3
    assert contains_dimension(builder._dimensions, inline_dim)
    assert contains_dimension(builder._dimensions, crossline_dim)
    assert contains_dimension(builder._dimensions, time_dim)
    assert len(added_dims) == 0

    #
    # Add non-existing dimension using its name is not allowed
    #
    with pytest.raises(ValueError, match="Dimension named 'offset' is not found"):
        builder._add_named_dimensions(["offset"])
        assert len(builder._dimensions) == 3
        assert contains_dimension(builder._dimensions, inline_dim)
        assert contains_dimension(builder._dimensions, crossline_dim)
        assert contains_dimension(builder._dimensions, time_dim)


def test__to_dictionary() -> None:
    """Test converting a BaseModel to a dictionary."""

    with pytest.raises(TypeError, match="Expected BaseModel, got datetime"):
        # This should raise an error because datetime is not a BaseModel
        to_dictionary(datetime.now()) 

    class SomeModel(StrictModel):
        count: int = Field(default=None, description="Samples count")
        samples: list[float] = Field(default_factory=list, description="Samples.")
        created: datetime = Field(default_factory=datetime.now, description="Creation time with TZ info.")

    m = SomeModel(  
        count = 3,
        samples = [1.0, 2.0, 3.0],
        created = datetime(2023, 10, 1, 12, 0, 0, tzinfo=None)
    )
    result = to_dictionary(m)
    assert isinstance(result, dict)
    assert result == {'count': 3, 'created': '2023-10-01T12:00:00', 'samples': [1.0, 2.0, 3.0]}


def test__make_VariableMetadata_from_list() -> None:
    """Test creating VariableMetadata from a strongly-typed list of AllUnits or UserAttributes."""

    units = AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT))
    attrs = UserAttributes(attributes={"MGA": 51, "UnitSystem": "Imperial"})
    meta_list=[units, attrs] 

    # TODO: I assume we do not want attribute.attribute in the contract:
    # 'metadata': {'unitsV1': {'length': 'm'}, 'attributes': {'attributes': {'MGA': 51}}}
    # TODO: Are multiple attributes allowed (I assume they are)?
    metadata = MDIODatasetBuilder._make_VariableMetadata_from_list(meta_list)
    assert isinstance(metadata, VariableMetadata)
    assert metadata.units_v1.length == "ft"
    assert metadata.attributes["MGA"] == 51 
    assert metadata.attributes["UnitSystem"] == "Imperial"

    with pytest.raises(TypeError, match="Unsupported metadata type: <class 'str'>"):
        meta_list = ["ft"]
        MDIODatasetBuilder._make_VariableMetadata_from_list(meta_list)

def test__make_VariableMetadata_from_dict() -> None:
    """Test creating VariableMetadata from a dictionary."""

    # TODO: What is the key for units: it unitsV1 or units_v1?
    # TODO: Are multiple attributes allowed (I assume they are)?
    # TODO: Do we validate the unit string supplied in dictionary parameters? What what if someone supplies ftUS instead of ft?
    meta_dict={"unitsV1": {"length": "ft"}, "attributes": {"MGA": 51, "UnitSystem": "Imperial"}}
    metadata = MDIODatasetBuilder._make_VariableMetadata_from_dict(meta_dict)
    assert isinstance(metadata, VariableMetadata)
    assert metadata.units_v1.length == "ft"
    assert metadata.attributes["MGA"] == 51
    assert metadata.attributes["UnitSystem"] == "Imperial"

    with pytest.raises(TypeError, match="Unsupported metadata key: 'units_v1'. Expected 'unitsV1', 'attributes', 'chunkGrid', or 'statsV1."):
        meta_dict = {"units_v1": "ft"}
        MDIODatasetBuilder._make_VariableMetadata_from_dict(meta_dict)

    with pytest.raises(TypeError, match="Invalid value for key 'attributes': 42. Expected a dictionary."):
        meta_dict = {"attributes": 42}
        MDIODatasetBuilder._make_VariableMetadata_from_dict(meta_dict)  

    # *** We currently do not validate the structure of the value dictionaries ***
    # Pass unit object with invalid structure
    # with pytest.raises(TypeError, match="Invalid value format for key 'unitsV1': {'length': 'm', 'time': 'sec'}. "):
    #     meta_dict1 = {"unitsV1": {"length": "m", "time": "sec"}}
    #     MDIODatasetBuilder._make_VariableMetadata_from_dict(meta_dict1)



