"""Unit tests for MDIO v1 factory."""

# TODO(BrianMichell, #535): Update this to use canonical factory functions.

from datetime import UTC
from datetime import datetime

import pytest
from pydantic import ValidationError

from mdio.core.v1._serializer import make_coordinate
from mdio.core.v1._serializer import make_dataset
from mdio.core.v1._serializer import make_dataset_metadata
from mdio.core.v1._serializer import make_named_dimension
from mdio.core.v1._serializer import make_variable
from mdio.core.v1.builder import write_mdio_metadata
from mdio.core.v1.factory import SCHEMA_TEMPLATE_MAP
from mdio.core.v1.factory import MDIOSchemaType
from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType


def test_make_toy_dataset() -> None:
    """Test that make_toy_dataset returns a Dataset object using the factory pattern."""
    # Create dataset using factory
    template = SCHEMA_TEMPLATE_MAP[MDIOSchemaType.SEISMIC_3D_POST_STACK_GENERIC]
    ds = template.create(
        name="campos_3d",
        shape=[256, 512, 384],  # inline, crossline, time
        header_fields={
            "cdp-x": "int32",
            "cdp-y": "int32",
            "elevation": "float16",
            "some_scalar": "float16",
        },
        create_coords=True,
        sample_format="float32",
        chunks=[128, 128, 128],
        z_units={"unitsV1": {"time": "ms"}},
        attributes={
            "textHeader": [
                "C01 .......................... ",
                "C02 .......................... ",
                "C03 .......................... ",
            ],
            "foo": "bar",
        },
    )

    # Print the JSON representation of the dataset schema
    print("\nDataset Schema JSON:")
    print(ds.model_dump_json(indent=2))

    write_mdio_metadata(ds, "test_toy_dataset.mdio")

    # Verify metadata
    assert ds.metadata.name == "campos_3d"
    assert ds.metadata.api_version == "1.0.0"
    assert ds.metadata.attributes == {
        "textHeader": [
            "C01 .......................... ",
            "C02 .......................... ",
            "C03 .......................... ",
        ],
        "foo": "bar",
    }

    # Verify variables, coordinates, and dimensions
    assert len(ds.variables) == 8  # noqa: PLR2004

    # Find seismic variable
    seismic = next(v for v in ds.variables if v.name == "seismic")
    assert seismic.data_type == ScalarType.FLOAT32
    assert seismic.dimensions[0].name == "inline"
    assert seismic.dimensions[1].name == "crossline"
    assert seismic.dimensions[2].name == "sample"
    assert seismic.compressor == Blosc(name="blosc", algorithm="zstd")

    # Find headers variable
    headers = next(v for v in ds.variables if v.name == "headers")
    assert isinstance(headers.data_type, StructuredType)
    assert len(headers.data_type.fields) == 4  # noqa: PLR2004
    assert headers.dimensions[0].name == "inline"
    assert headers.dimensions[1].name == "crossline"
    assert headers.compressor == Blosc(name="blosc")

    # Find trace mask
    mask = next(v for v in ds.variables if v.name == "trace_mask")
    assert mask.data_type == ScalarType.BOOL
    assert mask.dimensions[0].name == "inline"
    assert mask.dimensions[1].name == "crossline"
    assert mask.compressor == Blosc(name="blosc")

    # Find coordinates
    cdp_x = next(v for v in ds.variables if v.name == "cdp-x")
    assert cdp_x.data_type == ScalarType.FLOAT64
    assert cdp_x.dimensions[0].name == "inline"
    assert cdp_x.dimensions[1].name == "crossline"
    assert cdp_x.metadata.units_v1.length == "m"

    cdp_y = next(v for v in ds.variables if v.name == "cdp-y")
    assert cdp_y.data_type == ScalarType.FLOAT64
    assert cdp_y.dimensions[0].name == "inline"
    assert cdp_y.dimensions[1].name == "crossline"
    assert cdp_y.metadata.units_v1.length == "m"


def test_named_dimension_invalid_size() -> None:
    """Test that make_named_dimension raises a ValidationError for invalid size."""
    with pytest.raises(ValidationError):
        make_named_dimension("dim", 0)
    with pytest.raises(ValidationError):
        make_named_dimension("dim", -1)


def test_make_coordinate_invalid_types() -> None:
    """Test that make_coordinate raises a ValidationError for invalid types."""
    # dimensions must be a list of NamedDimension or str
    with pytest.raises(ValidationError):
        make_coordinate(name="coord", dimensions="notalist", data_type=ScalarType.FLOAT32)
    # data_type must be a valid ScalarType
    with pytest.raises(ValidationError):
        make_coordinate(name="coord", dimensions=["x"], data_type="notatype")


def test_make_variable_invalid_args() -> None:
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


def test_make_dataset_metadata_invalid_created_on() -> None:
    """Test that make_dataset_metadata raises a ValidationError for invalid created_on."""
    # created_on must be an aware datetime
    with pytest.raises(ValidationError):
        make_dataset_metadata(name="ds", api_version="1", created_on="not-a-date")


def test_make_dataset_invalid_variables_and_metadata_types() -> None:
    """Test that make_dataset raises a ValidationError."""
    ts = datetime.now(UTC)
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
