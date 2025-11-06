"""Unit tests for Seismic3DStreamerFieldRecordsTemplate."""

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
from mdio.builder.templates.seismic_3d_streamer_field import Seismic3DStreamerFieldRecordsTemplate

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)


DATASET_SIZE_MAP = {"sail_line": 1, "gun": 2, "shot_index": 128, "cable": 256, "channel": 12, "time": 1024}
DATASET_DTYPE_MAP = {"sail_line": "uint32", "gun": "uint8", "cable": "uint8", "channel": "uint16", "time": "int32"}
EXPECTED_COORDINATES = [
    "shot_point",
    "orig_field_record_num",
    "source_coord_x",
    "source_coord_y",
    "group_coord_x",
    "group_coord_y",
]


def _validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType, domain: str) -> None:
    """Validate the coordinate, headers, trace_mask variables in the dataset."""
    # Verify variables
    # 6 dim coords + 5 non-dim coords + 1 data + 1 trace mask + 1 headers = 14 variables
    assert len(dataset.variables) == 14

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
        if dim_name == "shot_index":
            continue

        validate_variable(
            dataset,
            name=dim_name,
            dims=[(dim_name, dim_size)],
            coords=[dim_name],
            dtype=ScalarType(DATASET_DTYPE_MAP[dim_name]),
        )

    # Verify non-dimension coordinate variables
    validate_variable(
        dataset,
        name="orig_field_record_num",
        dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k in ["sail_line", "gun", "shot_index"]],
        coords=["orig_field_record_num"],
        dtype=ScalarType.UINT32,
    )

    validate_variable(
        dataset,
        name="shot_point",
        dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k in ["sail_line", "gun", "shot_index"]],
        coords=["shot_point"],
        dtype=ScalarType.UINT32,
    )

    # Verify coordinate variables with units
    for coord_name in ["source_coord_x", "source_coord_y"]:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k in ["sail_line", "gun", "shot_index"]],
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER

    for coord_name in ["group_coord_x", "group_coord_y"]:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k != domain],
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER


class TestSeismic3DStreamerFieldRecordsTemplate:
    """Unit tests for Seismic3DStreamerFieldRecordsTemplate."""

    def test_configuration(self) -> None:
        """Unit tests for Seismic3DStreamerFieldRecordsTemplate."""
        t = Seismic3DStreamerFieldRecordsTemplate(data_domain="time")

        # Template attributes
        assert t.name == "StreamerFieldRecords3D"
        assert t._dim_names == ("sail_line", "gun", "shot_index", "cable", "channel", "time")
        assert t._physical_coord_names == ("source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y")
        assert t.full_chunk_shape == (1, 1, 16, 1, 32, 1024)

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == ()

        # Verify dataset attributes
        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyDimensionality": "3D", "gatherType": "common_source"}
        assert t.default_variable_name == "amplitude"

    def test_build_dataset(self, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DStreamerFieldRecordsTemplate build."""
        t = Seismic3DStreamerFieldRecordsTemplate(data_domain="time")
        t.add_units({"source_coord_x": UNITS_METER, "source_coord_y": UNITS_METER})  # spatial domain units
        t.add_units({"group_coord_x": UNITS_METER, "group_coord_y": UNITS_METER})  # spatial domain units
        t.add_units({"time": UNITS_SECOND})  # data domain units

        dataset = t.build_dataset("Survey3D", sizes=(1, 2, 128, 256, 12, 1024), header_dtype=structured_headers)

        assert dataset.metadata.name == "Survey3D"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["gatherType"] == "common_source"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, "time")

        # Verify seismic variable
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("sail_line", 1), ("gun", 2), ("shot_index", 128), ("cable", 256), ("channel", 12), ("time", 1024)],
            coords=[
                "shot_point",
                "orig_field_record_num",
                "source_coord_x",
                "source_coord_y",
                "group_coord_x",
                "group_coord_y",
            ],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.cname == BloscCname.zstd
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (1, 1, 16, 1, 32, 1024)
        assert seismic.metadata.stats_v1 is None
