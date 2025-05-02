"""Unit tests for the MDIO v1 template builder."""

import pytest
from datetime import datetime

from mdio.schema.dimension import NamedDimension
from mdio.schema.dtype import ScalarType
from mdio.schema.v1.template_builder import (
    TemplateBuilder,
    _BuilderState
)
from mdio.schema.v1.dataset import Dataset
from mdio.schema.v1.variable import Variable, Coordinate
from mdio.schema.compressors import Blosc, ZFP
from mdio.schema.dtype import StructuredType
from mdio.core.v1.xarray_constructor import to_mdio_zarr


def test_builder_initialization():
    """Test basic builder initialization."""
    builder = TemplateBuilder("test_dataset")
    assert builder.name == "test_dataset"
    assert builder.api_version == "1.0.0"
    assert isinstance(builder.created_on, datetime)
    assert len(builder._dimensions) == 0
    assert len(builder._coordinates) == 0
    assert len(builder._variables) == 0
    assert builder._state == _BuilderState.INITIAL


def test_dimension_builder_state():
    """Test dimension builder state transitions and functionality."""
    builder = TemplateBuilder("test_dataset")
    
    # First dimension should change state to HAS_DIMENSIONS and create a variable
    builder = builder.add_dimension("x", 100, long_name="X Dimension")
    assert builder._state == _BuilderState.HAS_DIMENSIONS
    assert len(builder._dimensions) == 1
    assert len(builder._variables) == 1
    assert builder._dimensions[0].name == "x"
    assert builder._dimensions[0].size == 100
    assert builder._variables[0].name == "x"
    assert builder._variables[0].long_name == "X Dimension"
    assert builder._variables[0].data_type == ScalarType.INT32
    assert builder._variables[0].dimensions[0].name == "x"

    # Adding another dimension should maintain state and create another variable
    builder = builder.add_dimension("y", 200, data_type=ScalarType.UINT32)
    assert builder._state == _BuilderState.HAS_DIMENSIONS
    assert len(builder._dimensions) == 2
    assert len(builder._variables) == 2
    assert builder._dimensions[1].name == "y"
    assert builder._dimensions[1].size == 200
    assert builder._variables[1].name == "y"
    assert builder._variables[1].data_type == ScalarType.UINT32
    assert builder._variables[1].dimensions[0].name == "y"


def test_dimension_with_metadata():
    """Test adding dimensions with custom metadata."""
    builder = TemplateBuilder("test_dataset")
    
    # Add dimension with custom metadata
    builder = builder.add_dimension(
        "depth",
        size=100,
        data_type=ScalarType.FLOAT32,
        metadata={"unitsV1": {"length": "m"}}
    )
    
    assert len(builder._variables) == 1
    depth_var = builder._variables[0]
    assert depth_var.name == "depth"
    assert depth_var.data_type == ScalarType.FLOAT32
    assert depth_var.metadata.units_v1.length == "m"


def test_coordinate_builder_state():
    """Test coordinate builder state transitions and functionality."""
    builder = TemplateBuilder("test_dataset")
    
    # Should not be able to add coordinates before dimensions
    with pytest.raises(ValueError, match="Must add at least one dimension before adding coordinates"):
        builder.add_coordinate("x_coord", dimensions=["x"])
    
    # Add dimensions first
    builder = builder.add_dimension("x", 100)
    builder = builder.add_dimension("y", 200)
    
    # Adding coordinate should change state to HAS_COORDINATES
    builder = builder.add_coordinate("x_coord", dimensions=["x"], long_name="X Coordinate")
    assert builder._state == _BuilderState.HAS_COORDINATES
    assert len(builder._coordinates) == 1
    assert builder._coordinates[0].name == "x_coord"
    assert builder._coordinates[0].long_name == "X Coordinate"
    assert builder._coordinates[0].dimensions[0].name == "x"

    # Adding another coordinate should maintain state
    builder = builder.add_coordinate("y_coord", dimensions=["y"])
    assert builder._state == _BuilderState.HAS_COORDINATES
    assert len(builder._coordinates) == 2
    assert builder._coordinates[1].name == "y_coord"
    assert builder._coordinates[1].dimensions[0].name == "y"


def test_variable_builder_state():
    """Test variable builder state transitions and functionality."""
    builder = TemplateBuilder("test_dataset")
    
    # Should not be able to add variables before dimensions
    with pytest.raises(ValueError, match="Must add at least one dimension before adding variables"):
        builder.add_variable("data", dimensions=["x"])
    
    # Add dimension first
    builder = builder.add_dimension("x", 100)
    
    # Adding variable should change state to HAS_VARIABLES
    builder = builder.add_variable("data", dimensions=["x"], long_name="Data Variable")
    assert builder._state == _BuilderState.HAS_VARIABLES
    assert len(builder._variables) == 2  # One for dimension, one for variable
    assert builder._variables[1].name == "data"
    assert builder._variables[1].long_name == "Data Variable"
    assert builder._variables[1].dimensions[0].name == "x"

    # Adding another variable should maintain state
    builder = builder.add_variable("data2", dimensions=["x"])
    assert builder._state == _BuilderState.HAS_VARIABLES
    assert len(builder._variables) == 3  # One for dimension, two for variables
    assert builder._variables[2].name == "data2"
    assert builder._variables[2].dimensions[0].name == "x"


def test_build_dataset():
    """Test building a complete dataset."""
    dataset = (TemplateBuilder("test_dataset")
        .add_dimension("x", 100)
        .add_dimension("y", 200)
        .add_coordinate("x_coord", dimensions=["x"])
        .add_coordinate("y_coord", dimensions=["y"])
        .add_variable("data", dimensions=["x", "y"], long_name="Test Data")
        .build())
    
    assert isinstance(dataset, Dataset)
    assert dataset.metadata.name == "test_dataset"
    # Two dimension variables + one data variable + two coordinate variables
    assert len(dataset.variables) == 5
    assert dataset.variables[0].name == "x"
    assert dataset.variables[1].name == "y"
    assert dataset.variables[2].name == "data"
    assert dataset.variables[2].long_name == "Test Data"
    assert len(dataset.variables[2].dimensions) == 2


def test_auto_naming():
    """Test automatic naming of coordinates and variables."""
    dataset = (TemplateBuilder("test_dataset")
        .add_dimension("x", 100)
        .add_coordinate()  # Should be named "coord_0"
        .add_coordinate()  # Should be named "coord_1"
        .add_variable()    # Should be named "var_0"
        .add_variable()    # Should be named "var_1"
        .build())
    
    assert dataset.variables[0].name == "x"  # Dimension variable
    assert dataset.variables[1].name == "var_0"
    assert dataset.variables[2].name == "var_1"


def test_default_dimensions():
    """Test that coordinates and variables use all dimensions by default."""
    dataset = (TemplateBuilder("test_dataset")
        .add_dimension("x", 100)
        .add_dimension("y", 200)
        .add_coordinate()  # Should use both x and y dimensions
        .add_variable()    # Should use both x and y dimensions
        .build())

    # Two dimension variables + one data variable + one coordinate variable
    assert len(dataset.variables) == 4
    assert dataset.variables[2].name == "var_0"
    assert len(dataset.variables[2].dimensions) == 2
    assert dataset.variables[2].dimensions[0].name == "x"
    assert dataset.variables[2].dimensions[1].name == "y"


def test_build_order_enforcement():
    """Test that the builder enforces the correct build order."""
    builder = TemplateBuilder("test_dataset")
    
    # Should not be able to add coordinates before dimensions
    with pytest.raises(ValueError, match="Must add at least one dimension before adding coordinates"):
        builder.add_coordinate("x_coord", dimensions=["x"])
    
    # Should not be able to add variables before dimensions
    with pytest.raises(ValueError, match="Must add at least one dimension before adding variables"):
        builder.add_variable("data", dimensions=["x"])
    
    # Should not be able to build without dimensions
    with pytest.raises(ValueError, match="Must add at least one dimension before building"):
        builder.build()


def test_toy_example():
    """Test building a toy dataset with multiple variables and attributes."""
    dataset = (TemplateBuilder("campos_3d", attributes={
        "textHeader": [
            "C01 .......................... ",
            "C02 .......................... ",
            "C03 .......................... "
        ],
        "foo": "bar"
    })
        # Add dimensions
        .add_dimension("inline", 256, data_type=ScalarType.UINT32)
        .add_dimension("crossline", 512, data_type=ScalarType.UINT32)
        .add_dimension("depth", 384, data_type=ScalarType.UINT32, metadata={"unitsV1": {"length": "m"}})
        
        # Add coordinates
        .add_coordinate("cdp-x", dimensions=["inline", "crossline"], metadata={"unitsV1": {"length": "m"}})
        .add_coordinate("cdp-y", dimensions=["inline", "crossline"], metadata={"unitsV1": {"length": "m"}})
        
        # Add image variable
        .add_variable(
            name="image",
            dimensions=["inline", "crossline", "depth"],
            data_type=ScalarType.FLOAT32,
            compressor=Blosc(algorithm="zstd"),
            coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
            metadata={
                "chunkGrid": {"name": "regular", "configuration": {"chunkShape": [128, 128, 128]}},
                "statsV1": {
                    "count": 100,
                    "sum": 1215.1,
                    "sumSquares": 125.12,
                    "min": 5.61,
                    "max": 10.84,
                    "histogram": {"binCenters": [1, 2], "counts": [10, 15]}
                },
                "attributes": {"fizz": "buzz"}
            }
        )
        
        # Add velocity variable
        .add_variable(
            name="velocity",
            dimensions=["inline", "crossline", "depth"],
            data_type=ScalarType.FLOAT16,
            coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
            metadata={
                "chunkGrid": {"name": "regular", "configuration": {"chunkShape": [128, 128, 128]}},
                "unitsV1": {"speed": "m/s"}
            }
        )
        
        # Add inline-optimized image variable
        .add_variable(
            name="image_inline",
            long_name="inline optimized version of 3d_stack",
            dimensions=["inline", "crossline", "depth"],
            data_type=ScalarType.FLOAT32,
            compressor=Blosc(algorithm="zstd"),
            coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
            metadata={
                "chunkGrid": {"name": "regular", "configuration": {"chunkShape": [4, 512, 512]}}
            }
        )
        
        # Add headers variable with structured dtype
        .add_variable(
            name="image_headers",
            dimensions=["inline", "crossline"],
            data_type=StructuredType(fields=[
                {"name": "cdp-x", "format": ScalarType.INT32},
                {"name": "cdp-y", "format": ScalarType.INT32},
                {"name": "elevation", "format": ScalarType.FLOAT16},
                {"name": "some_scalar", "format": ScalarType.FLOAT16}
            ]),
            coordinates=["inline", "crossline", "cdp-x", "cdp-y"],
            # metadata={"chunkGrid": {"name": "regular", "configuration": {"chunkShape": [128, 128]}}}
        )
        .build())

    # print(dataset.model_dump_json(indent=2))

    path = "test.mdio1"
    to_mdio_zarr(dataset, path)

    # Verify dataset structure
    assert dataset.metadata.name == "campos_3d"
    assert dataset.metadata.api_version == "1.0.0"
    assert dataset.metadata.attributes["foo"] == "bar"
    assert len(dataset.metadata.attributes["textHeader"]) == 3
    
    # Verify variables (including dimension variables)
    # 3 dimension variables + 4 data variables + 2 coordinate variables
    assert len(dataset.variables) == 9
    
    # Verify dimension variables
    inline_var = next(v for v in dataset.variables if v.name == "inline")
    assert inline_var.data_type == ScalarType.UINT32
    assert len(inline_var.dimensions) == 1
    assert inline_var.dimensions[0].name == "inline"
    
    depth_var = next(v for v in dataset.variables if v.name == "depth")
    assert depth_var.data_type == ScalarType.UINT32
    assert depth_var.metadata.units_v1.length == "m"
    
    # Verify image variable
    image = next(v for v in dataset.variables if v.name == "image")
    assert image.data_type == ScalarType.FLOAT32
    assert isinstance(image.compressor, Blosc)
    assert image.compressor.algorithm == "zstd"
    assert image.metadata.stats_v1.count == 100
    
    # Verify velocity variable
    velocity = next(v for v in dataset.variables if v.name == "velocity")
    assert velocity.data_type == ScalarType.FLOAT16
    assert velocity.compressor is None
    assert velocity.metadata.units_v1.speed == "m/s"
    
    # Verify image_inline variable
    image_inline = next(v for v in dataset.variables if v.name == "image_inline")
    assert image_inline.long_name == "inline optimized version of 3d_stack"
    assert isinstance(image_inline.compressor, Blosc)
    assert image_inline.compressor.algorithm == "zstd"
    
    # Verify image_headers variable
    headers = next(v for v in dataset.variables if v.name == "image_headers")
    assert isinstance(headers.data_type, StructuredType)
    assert len(headers.data_type.fields) == 4
    assert headers.data_type.fields[0].name == "cdp-x"
