"""Unit tests for Seismic3DObnSingleComponentGathersTemplate."""

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
from mdio.builder.templates.seismic_3d_obn_single_component import Seismic3DObnSingleComponentGathersTemplate

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)

DATASET_SIZE_MAP = {"receiver": 500, "shot_line": 10, "gun": 2, "shot_point": 200, "time": 4096}
DATASET_DTYPE_MAP = {
    "receiver": "uint32",
    "shot_line": "uint32",
    "gun": "uint8",
    "shot_point": "uint32",
    "time": "int32",
}
EXPECTED_COORDINATES = [
    "group_coord_x",
    "group_coord_y",
    "source_coord_x",
    "source_coord_y",
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

    # Verify receiver coordinate variables
    for coord_name in ["group_coord_x", "group_coord_y"]:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=[("receiver", DATASET_SIZE_MAP["receiver"])],
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify source coordinate variables
    shot_dims = [(k, v) for k, v in DATASET_SIZE_MAP.items() if k in ["shot_line", "gun", "shot_point"]]
    for coord_name in ["source_coord_x", "source_coord_y"]:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=shot_dims,
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER


class TestSeismic3DObnSingleComponentGathersTemplate:
    """Unit tests for Seismic3DObnSingleComponentGathersTemplate."""

    def test_configuration(self) -> None:
        """Test template configuration and attributes."""
        t = Seismic3DObnSingleComponentGathersTemplate(data_domain="time")

        assert t.name == "ObnSingleComponentGathers3D"
        assert t._dim_names == ("receiver", "shot_line", "gun", "shot_point", "time")
        assert t._physical_coord_names == (
            "group_coord_x",
            "group_coord_y",
            "source_coord_x",
            "source_coord_y",
        )
        assert t._logical_coord_names == ("orig_field_record_num",)
        assert t._var_chunk_shape == (16, 1, 2, 16, 4096)

        assert t._builder is None
        assert t._dim_sizes == ()

        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyType": "3D", "gatherType": "common_receiver"}
        assert t.default_variable_name == "amplitude"

    def test_chunk_size_calculation(self) -> None:
        """Test that chunk shape produces approximately 8 MiB chunks.

        The chunk shape (16, 1, 2, 16, 4096) produces:
        16 * 1 * 2 * 16 * 4096 = 2,097,152 samples.
        With float32 (4 bytes): 2,097,152 * 4 = 8,388,608 bytes = 8 MiB.
        """
        t = Seismic3DObnSingleComponentGathersTemplate(data_domain="time")

        chunk_shape = t.full_chunk_shape
        assert chunk_shape == (16, 1, 2, 16, 4096)

        samples_per_chunk = 1
        for dim_size in chunk_shape:
            samples_per_chunk *= dim_size

        bytes_per_chunk = samples_per_chunk * 4
        assert bytes_per_chunk == 8 * 1024 * 1024  # 8 MiB

    def test_isometric_chunking(self) -> None:
        """Test that receiver and shot_point have balanced chunk sizes for isometric access."""
        t = Seismic3DObnSingleComponentGathersTemplate(data_domain="time")

        chunk_shape = t._var_chunk_shape
        receiver_chunk = chunk_shape[0]  # receiver
        shot_point_chunk = chunk_shape[3]  # shot_point

        assert receiver_chunk == shot_point_chunk == 16

    def test_build_dataset(self, structured_headers: StructuredType) -> None:
        """Test building a complete dataset with the template."""
        t = Seismic3DObnSingleComponentGathersTemplate(data_domain="time")

        t.add_units({"group_coord_x": UNITS_METER, "group_coord_y": UNITS_METER})
        t.add_units({"source_coord_x": UNITS_METER, "source_coord_y": UNITS_METER})
        t.add_units({"time": UNITS_SECOND})

        sizes = tuple(DATASET_SIZE_MAP.values())
        dataset = t.build_dataset("ObnSurvey3D", sizes=sizes, header_dtype=structured_headers)

        assert dataset.metadata.name == "ObnSurvey3D"
        assert dataset.metadata.attributes["surveyType"] == "3D"
        assert dataset.metadata.attributes["gatherType"] == "common_receiver"
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
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (16, 1, 2, 16, 4096)
        assert seismic.metadata.stats_v1 is None

    def test_depth_domain(self, structured_headers: StructuredType) -> None:
        """Test building a dataset with depth domain."""
        t = Seismic3DObnSingleComponentGathersTemplate(data_domain="depth")

        assert t.trace_domain == "depth"
        assert t._dim_names == ("receiver", "shot_line", "gun", "shot_point", "depth")

        sizes = (100, 5, 2, 50, 2048)
        dataset = t.build_dataset("ObnSurveyDepth", sizes=sizes, header_dtype=structured_headers)

        assert dataset.metadata.name == "ObnSurveyDepth"

        depth_coord = next((v for v in dataset.variables if v.name == "depth"), None)
        assert depth_coord is not None
        assert depth_coord.dimensions[0].name == "depth"
        assert depth_coord.dimensions[0].size == 2048
