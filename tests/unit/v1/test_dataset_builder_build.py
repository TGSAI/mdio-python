"""Tests the schema v1 dataset_builder.build() public API."""

from mdio.builder.dataset_builder import MDIODatasetBuilder
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredField
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.v1.dataset import Dataset
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import SpeedUnitEnum

from .helpers import make_seismic_poststack_3d_acceptance_dataset
from .helpers import validate_variable


def test_build() -> None:
    """Test building a complete dataset."""
    dataset = (
        MDIODatasetBuilder("test_dataset")
        .add_dimension("inline", 100)
        .add_dimension("crossline", 200)
        .add_coordinate("inline", dimensions=("inline",), data_type=ScalarType.FLOAT64)
        .add_coordinate("crossline", dimensions=("crossline",), data_type=ScalarType.FLOAT64)
        .add_coordinate("x_coord", dimensions=("inline", "crossline"), data_type=ScalarType.FLOAT32)
        .add_coordinate("y_coord", dimensions=("inline", "crossline"), data_type=ScalarType.FLOAT32)
        .add_variable(
            "data",
            long_name="Test Data",
            dimensions=("inline", "crossline"),
            coordinates=("inline", "crossline", "x_coord", "y_coord"),
            data_type=ScalarType.FLOAT32,
        )
        .build()
    )

    assert isinstance(dataset, Dataset)
    assert dataset.metadata.name == "test_dataset"
    # 2 dim coord var + 2 non-dim coord var + 1 data variables = 5 variables
    assert len(dataset.variables) == 5
    assert next(v for v in dataset.variables if v.name == "inline") is not None
    assert next(v for v in dataset.variables if v.name == "crossline") is not None
    assert next(v for v in dataset.variables if v.name == "x_coord") is not None
    assert next(v for v in dataset.variables if v.name == "y_coord") is not None
    assert next(v for v in dataset.variables if v.name == "data") is not None


def test_build_seismic_poststack_3d_acceptance_dataset() -> None:  # noqa: PLR0915 Too many statements (57 > 50)
    """Test building a Seismic PostStack 3D Acceptance dataset."""
    dataset = make_seismic_poststack_3d_acceptance_dataset("Seismic")

    # Verify dataset structure
    assert dataset.metadata.name == "Seismic"
    assert dataset.metadata.api_version == "1.0.0a1"
    assert dataset.metadata.attributes["foo"] == "bar"
    assert len(dataset.metadata.attributes["textHeader"]) == 3

    # Verify variables (including dimension variables)
    # 3 dimension variables + 4 data variables + 2 coordinate variables
    assert len(dataset.variables) == 9

    # Verify dimension coordinate variables
    validate_variable(dataset, name="inline", dims=[("inline", 256)], coords=["inline"], dtype=ScalarType.UINT32)

    validate_variable(
        dataset,
        name="crossline",
        dims=[("crossline", 512)],
        coords=["crossline"],
        dtype=ScalarType.UINT32,
    )

    depth = validate_variable(dataset, name="depth", dims=[("depth", 384)], coords=["depth"], dtype=ScalarType.UINT32)
    assert depth.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify coordinate variables
    cdp_x = validate_variable(
        dataset,
        name="cdp_x",
        dims=[("inline", 256), ("crossline", 512)],
        coords=["cdp_x"],
        dtype=ScalarType.FLOAT32,
    )
    assert cdp_x.metadata.units_v1.length == LengthUnitEnum.METER

    cdp_y = validate_variable(
        dataset,
        name="cdp_y",
        dims=[("inline", 256), ("crossline", 512)],
        coords=["cdp_y"],
        dtype=ScalarType.FLOAT32,
    )
    assert cdp_y.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify data variables
    image = validate_variable(
        dataset,
        name="image",
        dims=[("inline", 256), ("crossline", 512), ("depth", 384)],
        coords=["cdp_x", "cdp_y"],
        dtype=ScalarType.FLOAT32,
    )
    assert image.metadata.units_v1 is None  # No units defined for image
    assert image.compressor.cname == BloscCname.zstd
    assert image.metadata.chunk_grid.configuration.chunk_shape == (128, 128, 128)
    assert image.metadata.stats_v1.count == 100

    velocity = validate_variable(
        dataset,
        name="velocity",
        dims=[("inline", 256), ("crossline", 512), ("depth", 384)],
        coords=["cdp_x", "cdp_y"],
        dtype=ScalarType.FLOAT16,
    )
    assert velocity.compressor is None
    assert velocity.metadata.chunk_grid.configuration.chunk_shape == (128, 128, 128)
    assert velocity.metadata.units_v1.speed == SpeedUnitEnum.METER_PER_SECOND

    image_inline = validate_variable(
        dataset,
        name="image_inline",
        dims=[("inline", 256), ("crossline", 512), ("depth", 384)],
        coords=["cdp_x", "cdp_y"],
        dtype=ScalarType.FLOAT32,
    )
    assert image_inline.long_name == "inline optimized version of 3d_stack"
    assert image_inline.compressor.cname == BloscCname.zstd
    assert image_inline.metadata.chunk_grid.configuration.chunk_shape == (4, 512, 512)

    # Verify image_headers variable
    headers = next(v for v in dataset.variables if v.name == "image_headers")
    assert isinstance(headers.data_type, StructuredType)
    assert len(headers.data_type.fields) == 4
    assert headers.data_type.fields[0].name == "cdp_x"

    headers = validate_variable(
        dataset,
        name="image_headers",
        dims=[("inline", 256), ("crossline", 512)],
        coords=["cdp_x", "cdp_y"],
        dtype=StructuredType(
            fields=[
                StructuredField(name="cdp_x", format=ScalarType.INT32),
                StructuredField(name="cdp_y", format=ScalarType.INT32),
                StructuredField(name="elevation", format=ScalarType.FLOAT16),
                StructuredField(name="some_scalar", format=ScalarType.FLOAT16),
            ]
        ),
    )
    assert headers.metadata.chunk_grid.configuration.chunk_shape == (128, 128)
