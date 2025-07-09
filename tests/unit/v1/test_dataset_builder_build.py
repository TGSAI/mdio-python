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
from mdio.schemas.v1.dataset_builder import _get_named_dimension
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
        .add_dimension("inline", 100)
        .add_dimension("crossline", 200)
        .add_coordinate("inline", dimensions=["inline"], data_type=ScalarType.FLOAT64) 
        .add_coordinate("crossline", dimensions=["crossline"], data_type=ScalarType.FLOAT64) 
        .add_coordinate("x_coord", dimensions=["inline", "crossline"])
        .add_coordinate("y_coord", dimensions=["inline", "crossline"])
        .add_variable("data", 
                      long_name="Test Data", 
                      dimensions=["inline", "crossline"], 
                      coordinates=["inline", "crossline", "x_coord", "y_coord"],
                      data_type=ScalarType.FLOAT32)
        .build()
    )

    # TODO: Expand this
    assert isinstance(dataset, Dataset)
    assert dataset.metadata.name == "test_dataset"
    # 2 dim coord var + 2 non-dim coord var + 1 data variables = 5 variables
    assert len(dataset.variables) == 5
    assert next(v for v in dataset.variables if v.name == "inline") is not None
    assert next(v for v in dataset.variables if v.name == "crossline") is not None
    assert next(v for v in dataset.variables if v.name == "x_coord") is not None
    assert next(v for v in dataset.variables if v.name == "y_coord") is not None
    assert next(v for v in dataset.variables if v.name == "data") is not None
    var_data = next(v for v in dataset.variables if v.name == "data")
    assert var_data is not None
    assert var_data.long_name == "Test Data"
    assert len(var_data.dimensions) == 2


def test_build_campos_3d() -> None: # noqa: PLR0915 Too many statements (57 > 50)
    """Test building a Campos 3D dataset with multiple variables and attributes."""
    dataset = make_campos_3d_dataset()

    # Verify dataset structure
    assert dataset.metadata.name == "campos_3d"
    assert dataset.metadata.api_version == "1.0.0a1"
    assert dataset.metadata.attributes["foo"] == "bar"
    assert len(dataset.metadata.attributes["textHeader"]) == 3

    # Verify variables (including dimension variables)
    # 3 dimension variables + 4 data variables + 2 coordinate variables
    assert len(dataset.variables) == 9

    # Verify dimension variables
    inline_var = next(v for v in dataset.variables if v.name == "inline")
    assert inline_var.data_type == ScalarType.UINT32
    # Dimension variables store dimensions as NamedDimension
    assert _get_named_dimension(inline_var.dimensions, "inline", 256)

    crossline_var = next(v for v in dataset.variables if v.name == "crossline")
    assert crossline_var.data_type == ScalarType.UINT32
    # Dimension variables store dimensions as NamedDimension
    assert _get_named_dimension(crossline_var.dimensions, "crossline", 512)

    depth_var = next(v for v in dataset.variables if v.name == "depth")
    assert depth_var.data_type == ScalarType.FLOAT64
    # Dimension variables store dimensions as NamedDimension
    assert _get_named_dimension(depth_var.dimensions, "depth", 384)
    assert depth_var.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify coordinate variables
    cdp_x = next(v for v in dataset.variables if v.name == "cdp-x")
    assert cdp_x.data_type == ScalarType.FLOAT32
    assert len(cdp_x.dimensions) == 2
    assert _get_named_dimension(cdp_x.dimensions, "inline", 256)
    assert _get_named_dimension(cdp_x.dimensions, "crossline", 512)
    assert cdp_x.metadata.units_v1.length == LengthUnitEnum.METER

    cdp_y = next(v for v in dataset.variables if v.name == "cdp-y")
    assert cdp_y.data_type == ScalarType.FLOAT32
    assert len(cdp_y.dimensions) == 2
    assert _get_named_dimension(cdp_y.dimensions, "inline", 256)
    assert _get_named_dimension(cdp_y.dimensions, "crossline", 512)
    assert cdp_y.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify image variable
    image = next(v for v in dataset.variables if v.name == "image")
    assert len(image.dimensions) == 3
    assert _get_named_dimension(image.dimensions, "inline", 256)
    assert _get_named_dimension(image.dimensions, "crossline", 512)
    assert _get_named_dimension(image.dimensions, "depth", 384)
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
    assert len(velocity.dimensions) == 3
    assert _get_named_dimension(velocity.dimensions, "inline", 256)
    assert _get_named_dimension(velocity.dimensions, "crossline", 512)
    assert _get_named_dimension(velocity.dimensions, "depth", 384)
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
    assert len(image_inline.dimensions) == 3
    assert _get_named_dimension(image_inline.dimensions, "inline", 256)
    assert _get_named_dimension(image_inline.dimensions, "crossline", 512)
    assert _get_named_dimension(image_inline.dimensions, "depth", 384)
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
    ds.add_dimension("inline", 256)
    ds.add_dimension("crossline", 512)
    ds.add_dimension("depth", 384)
    ds.add_coordinate("inline", dimensions=["inline"], data_type=ScalarType.UINT32) 
    ds.add_coordinate("crossline", dimensions=["crossline"], data_type=ScalarType.UINT32) 
    ds.add_coordinate("depth", dimensions=["depth"], data_type=ScalarType.FLOAT64, 
                      metadata_info=[
                          AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))
                      ])
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

# def test_edge_case_not_used_dimension():
#     builder = MDIODatasetBuilder("test_dataset")

#     builder.add_dimension("inline", 100)
#     builder.add_dimension("xline", 200)
#     builder.add_dimension("depth", 300)
#     builder.add_dimension("time", 600)

#     # Add 'dimension Coordinate' or 'index Coordinates', 
#     # the coordinates with the same name as a dimension, marked by *) on objects used in binary operations.
#     builder.add_coordinate("inline", dimensions=["inline"], data_type=ScalarType.FLOAT32)
#     builder.add_coordinate("xline", dimensions=["xline"], data_type=ScalarType.FLOAT32)
#     # No 'depth' dimension coordinate is provided

#     # Add 'non-dimension coordinates' before we can add a data variable
#     builder.add_coordinate("cdp-x", dimensions=["inline", "crossline"])
#     builder.add_coordinate("cdp-y", dimensions=["inline", "crossline"])

#     # Add data variable with full parameters
#     builder.add_variable("seismic", 
#                          dimensions=["inline", "crossline", "depth"],
#                          data_type=ScalarType.FLOAT64,
#                          coordinates=["cdp-x", "cdp-y"])   
    
#     # NOTE: The model has separate list list[Coordinate] | list[str] 
#     # It does not allow mixing names and NamedDimensions