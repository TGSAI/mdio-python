"""Unit tests for MDIO v1 factory."""

# TODO(BrianMichell): Update this to use canonical factory functions. 

from datetime import datetime
from datetime import timezone

import pytest
from pydantic import ValidationError

from mdio.core.v1._serializer import make_coordinate
from mdio.core.v1._serializer import make_dataset
from mdio.core.v1._serializer import make_dataset_metadata
from mdio.core.v1._serializer import make_named_dimension
from mdio.core.v1._serializer import make_variable
from mdio.schema.compressors import ZFP
from mdio.schema.compressors import Blosc
from mdio.schema.dimension import NamedDimension
from mdio.schema.dtype import ScalarType
from mdio.schema.dtype import StructuredType


def test_make_named_dimension():
    """Test that make_named_dimension returns a NamedDimension object."""
    dim = make_named_dimension("time", 42)
    assert isinstance(dim, NamedDimension)
    assert dim.name == "time"
    assert dim.size == 42


def test_make_coordinate_minimal():
    """Test that make_coordinate returns a Coordinate object."""
    dims = ["x"]
    coord = make_coordinate(name="x", dimensions=dims, data_type=ScalarType.FLOAT32)
    assert coord.name == "x"
    assert coord.dimensions == dims
    assert coord.data_type == ScalarType.FLOAT32
    assert coord.metadata is None


def test_make_variable_minimal():
    """Test that make_variable returns a Variable object."""
    var = make_variable(
        name="var",
        dimensions=["x"],
        data_type=ScalarType.FLOAT32,
        compressor=None,
    )
    assert var.name == "var"
    assert var.dimensions == ["x"]
    assert var.data_type == ScalarType.FLOAT32
    assert var.compressor is None
    assert var.coordinates is None
    assert var.metadata is None


def test_make_dataset_metadata_minimal():
    """Test that make_dataset_metadata returns a DatasetMetadata object."""
    ts = datetime.now(timezone.utc)
    meta = make_dataset_metadata(name="ds", api_version="1", created_on=ts)
    assert meta.name == "ds"
    assert meta.api_version == "1"
    assert meta.created_on == ts
    assert meta.attributes is None


def test_make_dataset_minimal():
    """Test that make_dataset returns a Dataset object."""
    var = make_variable(
        name="var",
        dimensions=["x"],
        data_type=ScalarType.FLOAT32,
        compressor=None,
    )
    ts = datetime.now(timezone.utc)
    meta = make_dataset_metadata(name="ds", api_version="1", created_on=ts)
    ds = make_dataset([var], meta)
    assert ds.variables == [var]
    assert ds.metadata == meta


def test_make_toy_dataset():
    """Test that make_toy_dataset returns a Dataset object."""
    # Define core dimensions
    inline = make_named_dimension("inline", 256)
    crossline = make_named_dimension("crossline", 512)
    depth = make_named_dimension("depth", 384)

    # Create dataset metadata
    created = datetime.fromisoformat("2023-12-12T15:02:06.413469-06:00")
    meta = make_dataset_metadata(
        name="campos_3d",
        api_version="1.0.0",
        created_on=created,
        attributes={
            "textHeader": [
                "C01 .......................... ",
                "C02 .......................... ",
                "C03 .......................... ",
            ],
            "foo": "bar",
        },
    )

    # Image variable
    image = make_variable(
        name="image",
        dimensions=[inline, crossline, depth],
        data_type=ScalarType.FLOAT32,
        compressor=Blosc(algorithm="zstd"),
        coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
        metadata={
            "chunkGrid": {
                "name": "regular",
                "configuration": {"chunkShape": [128, 128, 128]},
            },
            "statsV1": {
                "count": 100,
                "sum": 1215.1,
                "sumSquares": 125.12,
                "min": 5.61,
                "max": 10.84,
                "histogram": {"binCenters": [1, 2], "counts": [10, 15]},
            },
            "attributes": {"fizz": "buzz"},
        },
    )

    # Velocity variable
    velocity = make_variable(
        name="velocity",
        dimensions=[inline, crossline, depth],
        data_type=ScalarType.FLOAT16,
        compressor=None,
        coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
        metadata={
            "chunkGrid": {
                "name": "regular",
                "configuration": {"chunkShape": [128, 128, 128]},
            },
            "unitsV1": {"speed": "m/s"},
        },
    )

    # Inline-optimized image variable
    image_inline = make_variable(
        name="image_inline",
        dimensions=[inline, crossline, depth],
        data_type=ScalarType.FLOAT32,
        compressor=ZFP(mode="fixed_accuracy", tolerance=0.05),
        coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
        metadata={
            "chunkGrid": {
                "name": "regular",
                "configuration": {"chunkShape": [4, 512, 512]},
            }
        },
    )

    # Headers variable with structured dtype
    headers_dtype = StructuredType(
        fields=[
            {"name": "cdp-x", "format": ScalarType.INT32},
            {"name": "cdp-y", "format": ScalarType.INT32},
            {"name": "elevation", "format": ScalarType.FLOAT16},
            {"name": "some_scalar", "format": ScalarType.FLOAT16},
        ]
    )
    image_headers = make_variable(
        name="image_headers",
        dimensions=[inline, crossline],
        data_type=headers_dtype,
        compressor=None,
        coordinates=["inline", "crossline", "cdp-x", "cdp-y"],
        metadata={
            "chunkGrid": {
                "name": "regular",
                "configuration": {"chunkShape": [128, 128]},
            }
        },
    )

    # Standalone dimension arrays
    # Tests that we don't need to pass a compressor.
    inline_var = make_variable(
        name="inline", dimensions=[inline], data_type=ScalarType.UINT32
    )
    # Tests that we can still pass it explicitly.
    crossline_var = make_variable(
        name="crossline",
        dimensions=[crossline],
        data_type=ScalarType.UINT32,
        compressor=None,
    )
    depth_var = make_variable(
        name="depth",
        dimensions=[depth],
        data_type=ScalarType.UINT32,
        metadata={"unitsV1": {"length": "m"}},
    )
    cdp_x = make_variable(
        name="cdp-x",
        dimensions=[inline, crossline],
        data_type=ScalarType.FLOAT32,
        metadata={"unitsV1": {"length": "m"}},
    )
    cdp_y = make_variable(
        name="cdp-y",
        dimensions=[inline, crossline],
        data_type=ScalarType.FLOAT32,
        metadata={"unitsV1": {"length": "m"}},
    )

    # Compose full dataset
    ds = make_dataset(
        [
            image,
            velocity,
            image_inline,
            image_headers,
            inline_var,
            crossline_var,
            depth_var,
            cdp_x,
            cdp_y,
        ],
        meta,
    )

    assert ds.metadata == meta
    assert len(ds.variables) == 9
    assert ds.variables[0] == image
    assert ds.variables[1] == velocity
    assert ds.variables[2] == image_inline
    assert ds.variables[3] == image_headers
    assert ds.variables[4] == inline_var
    assert ds.variables[5] == crossline_var
    assert ds.variables[6] == depth_var
    assert ds.variables[7] == cdp_x
    assert ds.variables[8] == cdp_y


def test_named_dimension_invalid_size():
    """Test that make_named_dimension raises a ValidationError for invalid size."""
    with pytest.raises(ValidationError):
        make_named_dimension("dim", 0)
    with pytest.raises(ValidationError):
        make_named_dimension("dim", -1)


def test_make_coordinate_invalid_types():
    """Test that make_coordinate raises a ValidationError for invalid types."""
    # dimensions must be a list of NamedDimension or str
    with pytest.raises(ValidationError):
        make_coordinate(
            name="coord", dimensions="notalist", data_type=ScalarType.FLOAT32
        )
    # data_type must be a valid ScalarType
    with pytest.raises(ValidationError):
        make_coordinate(name="coord", dimensions=["x"], data_type="notatype")


def test_make_variable_invalid_args():
    """Test that make_variable raises a ValidationError for invalid types."""
    # compressor must be Blosc, ZFP or None
    with pytest.raises(ValidationError):
        make_variable(
            name="var",
            dimensions=["x"],
            data_type=ScalarType.FLOAT32,
            compressor="notacompressor",
        )
    # metadata dict must match VariableMetadata schema
    with pytest.raises(ValidationError):
        make_variable(
            name="var",
            dimensions=["x"],
            data_type=ScalarType.FLOAT32,
            compressor=None,
            metadata={"foo": "bar"},
        )


def test_make_dataset_metadata_invalid_created_on():
    """Test that make_dataset_metadata raises a ValidationError for invalid created_on."""
    # created_on must be an aware datetime
    with pytest.raises(ValidationError):
        make_dataset_metadata(name="ds", api_version="1", created_on="not-a-date")


def test_make_dataset_invalid_variables_and_metadata_types():
    """Test that make_dataset raises a ValidationError."""
    ts = datetime.now(timezone.utc)
    meta = make_dataset_metadata(name="ds", api_version="1", created_on=ts)
    var = make_variable(
        name="var",
        dimensions=["x"],
        data_type=ScalarType.FLOAT32,
        compressor=None,
    )
    # variables must be a list of Variable objects
    with pytest.raises(ValidationError):
        make_dataset(variables="notalist", metadata=meta)
    # metadata must be a DatasetMetadata instance
    with pytest.raises(ValidationError):
        make_dataset(variables=[var], metadata={"foo": "bar"})
