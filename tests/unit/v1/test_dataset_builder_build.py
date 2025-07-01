# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 dataset_builder.add_coordinate() public API."""

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import pytest

from mdio.schemas import builder
from mdio.schemas.chunk_grid import RegularChunkGrid, RegularChunkShape
from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType, StructuredType
from mdio.schemas.metadata import ChunkGridMetadata, UserAttributes
from mdio.schemas.v1.dataset import Dataset, DatasetInfo, DatasetMetadata
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.dataset_builder import _BuilderState
from mdio.schemas.v1.stats import CenteredBinHistogram, StatisticsMetadata, SummaryStatistics
from mdio.schemas.v1.units import AllUnits, SpeedUnitEnum, SpeedUnitModel
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.variable import VariableMetadata


def test_build() -> None:
    """Test adding coordinates. Check the state transition and validate required parameters."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder._state == _BuilderState.INITIAL

    builder.add_dimension("x", 100)
    builder.add_dimension("y", 100)

    builder.build()


def test_build_dataset() -> None:
    """Test building a complete dataset."""
    dataset = (
        MDIODatasetBuilder("test_dataset")
        .add_dimension("x", 100)
        .add_dimension("y", 200)
        .add_coordinate("x_coord", dimensions=["x"])
        .add_coordinate("y_coord", dimensions=["y"])
        .add_variable("data", dimensions=["x", "y"], long_name="Test Data")
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

# def test_build_dataset_bad_example() -> None:
#     builder = MDIODatasetBuilder("Bad example")
#     builder.add_dimension("inline", 256)
#     builder.add_dimension("crossline", 512)
#     builder.add_dimension("depth", 384)
#     builder.add_variable(name="image", dimensions=["inline", "crossline", "depth"])
#     dataset = builder.build()

#     json_str = dataset.model_dump_json()
#     file_path = os.path.join(os.path.dirname(__file__), "bad_example.json")
#     with open(file_path, 'w') as f:
#         f.write(json_str)

def test_build_campos_3d(tmp_path: Path) -> None:
    """Test building a toy dataset with multiple variables and attributes."""
    dataset = make_campos_3d_dataset()

    # Verify dataset structure
    assert dataset.metadata.name == "campos_3d"
    assert dataset.metadata.api_version == "1.0.0"
    assert dataset.metadata.attributes["foo"] == "bar"
    assert len(dataset.metadata.attributes["textHeader"]) == 3

    # Verify variables (including dimension variables)
    # 3 dimension variables + 4 data variables + 2 coordinate variables
    assert len(dataset.variables) == 9  # noqa: PLR2004

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
    image_inline = next(
        v for v in dataset.variables if v.name == "image_inline")
    assert image_inline.long_name == "inline optimized version of 3d_stack"
    assert isinstance(image_inline.compressor, Blosc)
    assert image_inline.compressor.algorithm == "zstd"

    # Verify image_headers variable
    headers = next(v for v in dataset.variables if v.name == "image_headers")
    assert isinstance(headers.data_type, StructuredType)
    assert len(headers.data_type.fields) == 4
    assert headers.data_type.fields[0].name == "cdp-x"

# def test_build_campos_3d_contract(tmp_path: Path) -> None:
#     '''Test building campos_3d dataset and converting to JSON schema format.'''
#     dataset = make_campos_3d_dataset()
#     # json_str = dataset.model_dump_json()
#     # json_sorted_str = json.dumps(json.loads(json_str), sort_keys=True)
#     m_dict = dataset.model_dump(mode="json", by_alias=True)
#     json_sorted_str = json.dumps(m_dict, sort_keys=True)

#     sorted_contract = load_campos_3d_contract()

#     save(json_sorted_str, sorted_contract)

# def save(model: str, contract: str) -> None:
#     model_file_path = os.path.join(os.path.dirname(__file__), "campos_3d_model.json")
#     contract_file_path = os.path.join(os.path.dirname(__file__), "campos_3d_contract.json")
#     with open(model_file_path, 'w') as f:
#         f.write(model)
#     with open(contract_file_path, 'w') as f:
#         f.write(contract)

# def load_campos_3d_contract():
#     file_path = os.path.join(os.path.dirname(__file__), 'test_data/campos_3d_contract.json')
#     with open(file_path, 'r') as f:
#         contract = json.load(f)
#     assert contract is not None
#     return json.dumps(contract, sort_keys=True)

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
        # metadata={"unitsV1": {"length": "m"}},
    )
    # Add image variable
    ds.add_variable(
        name="image",
        dimensions=["inline", "crossline", "depth"],
        data_type=ScalarType.FLOAT32,
        compressor=Blosc(algorithm="zstd"),
        coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
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
        coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
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
        coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
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
        coordinates=["inline", "crossline", "cdp-x", "cdp-y"],
    )
    return ds.build()
