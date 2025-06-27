# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 Variable public API."""

import pytest
from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset_builder import _BuilderState, MDIODatasetBuilder
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.variable import VariableMetadata

def test_dataset_builder_add_variable() -> None:
    """Test adding variable. Check the state transition and validate required parameters.."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder._state == _BuilderState.INITIAL
    
    with pytest.raises(ValueError, match="Must add at least one dimension before adding variables"):
        builder.add_variable("amplitude", dimensions=["speed"])

    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)
    builder.add_dimension("depth", 100)

    bad_name = None
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_variable(bad_name, dimensions=["speed"])
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_variable("", dimensions=["speed"])
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_variable("amplitude", dimensions=None)
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_variable("amplitude", dimensions=[])

    builder.add_variable("seismic_amplitude", dimensions=["inline", "crossline", "depth"])
    assert builder._state == _BuilderState.HAS_VARIABLES
    assert len(builder._dimensions) == 3 
    assert len(builder._variables) == 4 
    assert len(builder._coordinates) == 0  

def test_add_variable_with_defaults() -> None:
    """Test adding variable with default arguments."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 100)
    builder.add_dimension("depth", 100)
    # Add variable using defaults
    builder.add_variable("seismic_amplitude", dimensions=["inline", "crossline", "depth"])
    assert len(builder._dimensions) == 3 
    assert len(builder._variables) == 4 
    assert len(builder._coordinates) == 0 
    var0 = next((e for e in builder._variables if e.name == "seismic_amplitude"), None)
    assert var0 is not None
    # NOTE: add_variable() stores dimensions as names
    assert set(var0.dimensions) == set(["inline", "crossline", "depth"])
    assert var0.long_name is None               # Default value
    assert var0.data_type == ScalarType.FLOAT32 # Default value
    assert var0.compressor is None              # Default value
    assert var0.coordinates is None             # Default value
    assert var0.metadata is None                # Default value

# def test__make_variable_metadata() -> None:
#     """Test creating VariableMetadata from a dictionary."""

#     meta_dict = None
#     metadata = make_variable_metadata_list(meta_dict)
#     assert metadata is None

#     meta_dict = {}
#     metadata = make_variable_metadata_list(meta_dict)
#     assert metadata is None

#     meta_dict = {
#         "unitsV1": {"length": "m"},
#         "attributes": {"MGA": 51},
#         "chunkGrid": {"name": "regular", "configuration": {"chunkShape": [20]}},
#         "statsV1": {
#             "count": 100,
#             "sum": 1215.1,
#             "sumSquares": 125.12,
#             "min": 5.61,
#             "max": 10.84,
#             "histogram": {"binCenters": [1, 2], "counts": [10, 15]},
#         },
#     }
#     metadata = make_variable_metadata_list(meta_dict)
#     assert isinstance(metadata, VariableMetadata)
#     assert metadata.units_v1.length == "m"
#     assert metadata.attributes["MGA"] == 51  
#     assert metadata.chunk_grid.name == "regular"
#     assert metadata.chunk_grid.configuration.chunk_shape == [20]  
#     assert metadata.stats_v1.count == 100  
#     assert metadata.stats_v1.sum == 1215.1  
#     assert metadata.stats_v1.sum_squares == 125.12  
#     assert metadata.stats_v1.min == 5.61  
#     assert metadata.stats_v1.max == 10.84  
#     assert metadata.stats_v1.histogram.bin_centers == [1, 2]  
#     assert metadata.stats_v1.histogram.counts == [10, 15]  

#     # NOTE: the v1 schema has 'units_v1' property, but pydantic requires 'unitsV1' for CamelCaseStrictModel
#     meta_dict = {"units_v1": {"length": "m"}}
#     err_msg = (
#         "Unsupported metadata key: 'units_v1'. "
#         "Expected 'unitsV1', 'attributes', 'chunkGrid', or 'statsV1."
#     )
#     with pytest.raises(TypeError, match=err_msg):
#         make_variable_metadata_list(meta_dict)

#     meta_dict = {"attributes": 42}
#     with pytest.raises(
#         TypeError, match="Invalid value for key 'attributes': 42. Expected a dictionary."
#     ):
#         make_variable_metadata_list(meta_dict)

#     # *** We currently do not validate the structure of the value dictionaries ***
#     # Pass unit object with invalid structure
#     # with pytest.raises(
#     # TypeError, match="Invalid value format for key 'unitsV1': {'length': 'm', 'time': 'sec'}. "):
#     #     meta_dict1 = {"unitsV1": {"length": "m", "time": "sec"}}
#     #     _make_VariableMetadata_from_dict(meta_dict1)
