# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 dataset_builder.add_coordinate() public API."""


from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.chunk_grid import RegularChunkShape
from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.dataset_builder import _get_dimension
from mdio.schemas.v1.stats import CenteredBinHistogram
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.stats import SummaryStatistics
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.units import SpeedUnitEnum
from mdio.schemas.v1.units import SpeedUnitModel


def test_build() -> None:
    """Test building a complete dataset."""
    dataset = (
        MDIODatasetBuilder("test_dataset")
        .add_dimension("x", 100)
        .add_dimension("y", 200)
        .add_coordinate("x_coord", dimensions=["x"])
        .add_coordinate("y_coord", dimensions=["y"])
        .add_variable("data", dimensions=["x", "y"], 
                      long_name="Test Data", 
                      data_type=ScalarType.FLOAT32)
        .build()
    )
    assert isinstance(dataset, Dataset)
    assert dataset.metadata.name == "test_dataset"
    # 2 dimension variables + 1 data variable + 2 coordinate variables
    assert len(dataset.variables) == 5
    assert next(v for v in dataset.variables if v.name == "x") is not None
    assert next(v for v in dataset.variables if v.name == "y") is not None
    var_data = next(v for v in dataset.variables if v.name == "data")
    assert var_data is not None
    assert var_data.long_name == "Test Data"
    assert len(var_data.dimensions) == 2


def test_build_campos_3d() -> None: # noqa: PLR0915 Too many statements (57 > 50)
    """Test building a Campos 3D dataset with multiple variables and attributes."""
    dataset = make_campos_3d_dataset()

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
    # Dimension variables store dimensions as NamedDimension
    assert _get_dimension(inline_var.dimensions, "inline", 256)

    crossline_var = next(v for v in dataset.variables if v.name == "crossline")
    assert crossline_var.data_type == ScalarType.UINT32
    # Dimension variables store dimensions as NamedDimension
    assert _get_dimension(crossline_var.dimensions, "crossline", 512)

    depth_var = next(v for v in dataset.variables if v.name == "depth")
    assert depth_var.data_type == ScalarType.UINT32
    # Dimension variables store dimensions as NamedDimension
    assert _get_dimension(depth_var.dimensions, "depth", 384)
    assert depth_var.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify coordinate variables
    cdp_x = next(v for v in dataset.variables if v.name == "cdp-x")
    assert cdp_x.data_type == ScalarType.FLOAT32
    # Coordinates variables store dimensions as names
    assert set(cdp_x.dimensions) == {"inline", "crossline"}
    assert cdp_x.metadata.units_v1.length == LengthUnitEnum.METER

    cdp_y = next(v for v in dataset.variables if v.name == "cdp-y")
    assert cdp_y.data_type == ScalarType.FLOAT32
    # Coordinates variables store dimensions as names
    assert set(cdp_y.dimensions) == {"inline", "crossline"}
    assert cdp_y.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify image variable
    image = next(v for v in dataset.variables if v.name == "image")
    assert set(image.dimensions) == {"inline", "crossline", "depth"}
    assert image.data_type == ScalarType.FLOAT32
    assert isinstance(image.compressor, Blosc)
    assert image.compressor.algorithm == "zstd"
    # Other variables store dimensions as names
    assert set(image.coordinates) == {"cdp-x", "cdp-y"}
    assert isinstance(image.metadata.chunk_grid, RegularChunkGrid)
    assert image.metadata.chunk_grid.configuration.chunk_shape == [128, 128, 128]
    assert isinstance(image.metadata.stats_v1, SummaryStatistics)   
    assert image.metadata.stats_v1.count == 100

    # Verify velocity variable
    velocity = next(v for v in dataset.variables if v.name == "velocity")
    assert set(velocity.dimensions) == {"inline", "crossline", "depth"}
    assert velocity.data_type == ScalarType.FLOAT16
    assert velocity.compressor is None
    # Other variables store dimensions as names
    assert set(velocity.coordinates) == {"cdp-x", "cdp-y"}
    assert isinstance(velocity.metadata.chunk_grid, RegularChunkGrid)
    assert velocity.metadata.chunk_grid.configuration.chunk_shape == [128, 128, 128]
    assert isinstance(velocity.metadata.units_v1, SpeedUnitModel)
    assert velocity.metadata.units_v1.speed == SpeedUnitEnum.METER_PER_SECOND

    # Verify image_inline variable
    image_inline = next(
        v for v in dataset.variables if v.name == "image_inline")
    assert image_inline.long_name == "inline optimized version of 3d_stack"
    assert set(image_inline.dimensions) == {"inline", "crossline", "depth"}
    assert image_inline.data_type == ScalarType.FLOAT32
    assert isinstance(image_inline.compressor, Blosc)
    assert image_inline.compressor.algorithm == "zstd"
    assert set(image_inline.coordinates) == {"cdp-x", "cdp-y"}
    assert isinstance(image_inline.metadata.chunk_grid, RegularChunkGrid)
    assert image_inline.metadata.chunk_grid.configuration.chunk_shape == [4, 512, 512]

    # Verify image_headers variable
    headers = next(v for v in dataset.variables if v.name == "image_headers")
    assert isinstance(headers.data_type, StructuredType)
    assert len(headers.data_type.fields) == 4
    assert headers.data_type.fields[0].name == "cdp-x"

def make_campos_3d_dataset() -> Dataset:
    """Create in-memory campos_3d dataset."""
    ds = MDIODatasetBuilder(
        "campos_3d",
        attributes=UserAttributes(attributes={
            "textHeader": [
                "C01 .......................... ",
                "C02 .......................... ",
                "C03 .......................... ",
            ],
            "foo": "bar"
        }))

    # Add dimensions
    ds.add_dimension("inline", 256, var_data_type=ScalarType.UINT32)
    ds.add_dimension("crossline", 512, var_data_type=ScalarType.UINT32)
    ds.add_dimension("depth", 384, var_data_type=ScalarType.UINT32,
                     var_metadata_info=[
                         AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))]
                     )
    # Add coordinates
    ds.add_coordinate(
        "cdp-x",
        dimensions=["inline", "crossline"],
        metadata_info=[
            AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))]
    )
    ds.add_coordinate(
        "cdp-y",
        dimensions=["inline", "crossline"],
        metadata_info=[
            AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))]
    )
    # Add image variable
    ds.add_variable(
        name="image",
        dimensions=["inline", "crossline", "depth"],
        data_type=ScalarType.FLOAT32,
        compressor=Blosc(algorithm="zstd"),
        coordinates=["cdp-x", "cdp-y"],
        metadata_info=[
            ChunkGridMetadata(
                chunk_grid=RegularChunkGrid(
                    configuration=RegularChunkShape(chunk_shape=[128, 128, 128]))
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
            UserAttributes(
                attributes={"fizz": "buzz", "UnitSystem": "Canonical"}),
        ])
    # Add velocity variable
    ds.add_variable(
        name="velocity",
        dimensions=["inline", "crossline", "depth"],
        data_type=ScalarType.FLOAT16,
        coordinates=["cdp-x", "cdp-y"],
        metadata_info=[
            ChunkGridMetadata(
                chunk_grid=RegularChunkGrid(
                    configuration=RegularChunkShape(chunk_shape=[128, 128, 128]))
            ),
            AllUnits(units_v1=SpeedUnitModel(
                speed=SpeedUnitEnum.METER_PER_SECOND)),
        ],
    )
    # Add inline-optimized image variable
    ds.add_variable(
        name="image_inline",
        long_name="inline optimized version of 3d_stack",
        dimensions=["inline", "crossline", "depth"],
        data_type=ScalarType.FLOAT32,
        compressor=Blosc(algorithm="zstd"),
        coordinates=["cdp-x", "cdp-y"],
        metadata_info=[
            ChunkGridMetadata(
                chunk_grid=RegularChunkGrid(
                    configuration=RegularChunkShape(chunk_shape=[4, 512, 512]))
            )]
    )
    # Add headers variable with structured dtype
    ds.add_variable(
        name="image_headers",
        dimensions=["inline", "crossline"],
        data_type=StructuredType(
            fields=[
                {"name": "cdp-x", "format": ScalarType.INT32},
                {"name": "cdp-y", "format": ScalarType.INT32},
                {"name": "elevation", "format": ScalarType.FLOAT16},
                {"name": "some_scalar", "format": ScalarType.FLOAT16},
            ]
        ),
        coordinates=["cdp-x", "cdp-y"],
    )
    return ds.build()
