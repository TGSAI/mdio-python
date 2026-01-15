"""Unit tests for Seismic3DShotReceiverLineGathersTemplate."""

from tests.unit.v1.helpers import validate_variable

from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.v1.dataset import Dataset
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.schemas.v1.units import TimeUnitEnum
from mdio.builder.schemas.v1.units import TimeUnitModel
from mdio.builder.templates.seismic_3d_shot_receiver_line import Seismic3DShotReceiverLineGathersTemplate

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)

DATASET_SIZE_MAP = {
    "shot_line": 10,
    "shot_point": 200,
    "receiver_line": 50,
    "receiver": 100,
    "time": 4096,
}
DATASET_DTYPE_MAP = {
    "shot_line": "uint32",
    "shot_point": "uint32",
    "receiver_line": "uint32",
    "receiver": "uint32",
    "time": "int32",
}
EXPECTED_COORDINATES = [
    "source_coord_x",
    "source_coord_y",
    "receiver_coord_x",
    "receiver_coord_y",
    "orig_field_record_num",
]


def _validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType, domain: str) -> None:
    """Validate the coordinate, headers, trace_mask variables in the dataset."""
    # Verify variables
    # 5 dim coords + 5 non-dim coords + 1 data + 1 trace mask + 1 headers = 13 variables
    assert len(dataset.variables) == 13

    # Verify trace headers
    validate_variable(
        dataset,
        name="headers",
        dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k != domain],
        coords=EXPECTED_COORDINATES,
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k != domain],
        coords=EXPECTED_COORDINATES,
        dtype=ScalarType.BOOL,
    )

    # Verify dimension coordinate variables
    for dim_name, dim_size in DATASET_SIZE_MAP.items():
        validate_variable(
            dataset,
            name=dim_name,
            dims=[(dim_name, dim_size)],
            coords=[dim_name],
            dtype=ScalarType(DATASET_DTYPE_MAP[dim_name]),
        )

    # Verify source coordinate variables
    shot_dims = [(k, v) for k, v in DATASET_SIZE_MAP.items() if k in ["shot_line", "shot_point"]]
    for coord_name in ["source_coord_x", "source_coord_y"]:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=shot_dims,
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify receiver coordinate variables
    receiver_dims = [(k, v) for k, v in DATASET_SIZE_MAP.items() if k in ["receiver_line", "receiver"]]
    for coord_name in ["receiver_coord_x", "receiver_coord_y"]:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=receiver_dims,
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER


class TestSeismic3DShotReceiverLineGathersTemplate:
    """Unit tests for Seismic3DShotReceiverLineGathersTemplate."""

    def test_configuration(self) -> None:
        """Test template configuration and attributes."""
        t = Seismic3DShotReceiverLineGathersTemplate(data_domain="time")

        assert t.name == "ShotReceiverLineGathers3D"
        assert t._dim_names == ("shot_line", "shot_point", "receiver_line", "receiver", "time")
        assert t._physical_coord_names == (
            "source_coord_x",
            "source_coord_y",
            "receiver_coord_x",
            "receiver_coord_y",
        )
        assert t._logical_coord_names == ("orig_field_record_num",)
        assert t._var_chunk_shape == (1, 32, 1, 32, 2048)

        assert t._builder is None
        assert t._dim_sizes == ()

        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyDimensionality": "3D", "gatherType": "common_source"}
        assert t.default_variable_name == "amplitude"

    def test_chunk_size_calculation(self) -> None:
        """Test that chunk shape produces approximately 8 MiB chunks.

        The chunk shape (1, 32, 1, 32, 2048) produces:
        1 * 32 * 1 * 32 * 2048 = 2,097,152 samples.
        With float32 (4 bytes): 2,097,152 * 4 = 8,388,608 bytes = 8 MiB.
        """
        t = Seismic3DShotReceiverLineGathersTemplate(data_domain="time")

        chunk_shape = t.full_chunk_shape
        assert chunk_shape == (1, 32, 1, 32, 2048)

        samples_per_chunk = 1
        for dim_size in chunk_shape:
            samples_per_chunk *= dim_size

        bytes_per_chunk = samples_per_chunk * 4
        assert bytes_per_chunk == 8 * 1024 * 1024  # 8 MiB

    def test_isometric_chunking(self) -> None:
        """Test that shot_point and receiver have balanced chunk sizes for isometric access."""
        t = Seismic3DShotReceiverLineGathersTemplate(data_domain="time")

        chunk_shape = t._var_chunk_shape
        shot_point_chunk = chunk_shape[1]  # shot_point
        receiver_chunk = chunk_shape[3]  # receiver

        assert shot_point_chunk == receiver_chunk == 32

    def test_build_dataset(self, structured_headers: StructuredType) -> None:
        """Test building a complete dataset with the template."""
        t = Seismic3DShotReceiverLineGathersTemplate(data_domain="time")

        t.add_units({"source_coord_x": UNITS_METER, "source_coord_y": UNITS_METER})
        t.add_units({"receiver_coord_x": UNITS_METER, "receiver_coord_y": UNITS_METER})
        t.add_units({"time": UNITS_SECOND})

        sizes = tuple(DATASET_SIZE_MAP.values())
        dataset = t.build_dataset("LandSurvey3D", sizes=sizes, header_dtype=structured_headers)

        assert dataset.metadata.name == "LandSurvey3D"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["gatherType"] == "common_source"
        assert dataset.metadata.attributes["defaultVariableName"] == "amplitude"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, "time")

        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=list(DATASET_SIZE_MAP.items()),
            coords=EXPECTED_COORDINATES,
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.cname == BloscCname.zstd
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (1, 32, 1, 32, 2048)
        assert seismic.metadata.stats_v1 is None

    def test_depth_domain(self, structured_headers: StructuredType) -> None:
        """Test building a dataset with depth domain."""
        t = Seismic3DShotReceiverLineGathersTemplate(data_domain="depth")

        assert t.trace_domain == "depth"
        assert t._dim_names == ("shot_line", "shot_point", "receiver_line", "receiver", "depth")

        sizes = (5, 100, 20, 50, 2048)
        dataset = t.build_dataset("LandSurveyDepth", sizes=sizes, header_dtype=structured_headers)

        assert dataset.metadata.name == "LandSurveyDepth"

        depth_coord = next((v for v in dataset.variables if v.name == "depth"), None)
        assert depth_coord is not None
        assert depth_coord.dimensions[0].name == "depth"
        assert depth_coord.dimensions[0].size == 2048
