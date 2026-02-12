"""Unit tests for Seismic3DReceiverGathersTemplate."""

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
from mdio.builder.templates.seismic_3d_receiver_gathers import Seismic3DReceiverGathersTemplate

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)


EXPECTED_COORDINATES = [
    "receiver_x",
    "receiver_y",
    "shot_point",
    "source_coord_x",
    "source_coord_y",
]

DATASET_SIZE_MAP = {"receiver": 100, "shot_line": 10, "shot_index": 500, "time": 2048}


def _validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType) -> None:
    """Validate the coordinate, headers, trace_mask variables in the dataset."""
    dataset_dtype_map = {"receiver": "uint32", "shot_line": "uint32", "time": "int32"}

    # Verify variables
    # 3 dim coords (excluding shot_index) + 5 non-dim coords + 1 data + 1 trace mask + 1 headers = 11 variables
    assert len(dataset.variables) == 11

    # Verify trace headers
    validate_variable(
        dataset,
        name="headers",
        dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k != "time"],
        coords=EXPECTED_COORDINATES,
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k != "time"],
        coords=EXPECTED_COORDINATES,
        dtype=ScalarType.BOOL,
    )

    # Verify dimension coordinate variables (excluding shot_index which is calculated)
    for dim_name, dim_size in DATASET_SIZE_MAP.items():
        if dim_name == "shot_index":
            continue

        validate_variable(
            dataset,
            name=dim_name,
            dims=[(dim_name, dim_size)],
            coords=[dim_name],
            dtype=ScalarType(dataset_dtype_map[dim_name]),
        )

    # Verify non-dimension coordinate variables - receiver coordinates
    receiver_x = validate_variable(
        dataset,
        name="receiver_x",
        dims=[("receiver", DATASET_SIZE_MAP["receiver"])],
        coords=["receiver_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert receiver_x.metadata.units_v1.length == LengthUnitEnum.METER

    receiver_y = validate_variable(
        dataset,
        name="receiver_y",
        dims=[("receiver", DATASET_SIZE_MAP["receiver"])],
        coords=["receiver_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert receiver_y.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify shot_point coordinate (logical)
    validate_variable(
        dataset,
        name="shot_point",
        dims=[
            ("shot_line", DATASET_SIZE_MAP["shot_line"]),
            ("shot_index", DATASET_SIZE_MAP["shot_index"]),
        ],
        coords=["shot_point"],
        dtype=ScalarType.UINT32,
    )

    # Verify source coordinate variables
    for coord_name in ["source_coord_x", "source_coord_y"]:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=[
                ("shot_line", DATASET_SIZE_MAP["shot_line"]),
                ("shot_index", DATASET_SIZE_MAP["shot_index"]),
            ],
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER


class TestSeismic3DReceiverGathersTemplate:
    """Unit tests for Seismic3DReceiverGathersTemplate."""

    def test_configuration(self) -> None:
        """Test template configuration and attributes."""
        t = Seismic3DReceiverGathersTemplate()

        # Template attributes
        assert t.name == "ReceiverGathers3D"
        assert t._dim_names == ("receiver", "shot_line", "shot_index", "time")
        assert t._calculated_dims == ("shot_index",)
        assert t._physical_coord_names == ("receiver_x", "receiver_y", "source_coord_x", "source_coord_y")
        assert t._logical_coord_names == ("shot_point",)
        assert t.full_chunk_shape == (1, 1, 512, 4096)

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == ()

        # Verify dataset attributes
        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyType": "3D", "gatherType": "receiver_gathers"}
        assert t.default_variable_name == "amplitude"

    def test_chunk_size_calculation(self) -> None:
        """Test that chunk shape produces approximately 8 MiB chunks.

        The chunk shape (1, 1, 512, 4096) produces:
        1 * 1 * 512 * 4096 = 2,097,152 samples.
        With float32 (4 bytes): 2,097,152 * 4 = 8,388,608 bytes = 8 MiB.
        """
        t = Seismic3DReceiverGathersTemplate()

        chunk_shape = t.full_chunk_shape
        assert chunk_shape == (1, 1, 512, 4096)

        samples_per_chunk = 1
        for dim_size in chunk_shape:
            samples_per_chunk *= dim_size

        bytes_per_chunk = samples_per_chunk * 4
        assert bytes_per_chunk == 8 * 1024 * 1024  # 8 MiB

    def test_build_dataset(self, structured_headers: StructuredType) -> None:
        """Test building a complete dataset with the template."""
        t = Seismic3DReceiverGathersTemplate()
        t.add_units({"receiver_x": UNITS_METER, "receiver_y": UNITS_METER})
        t.add_units({"source_coord_x": UNITS_METER, "source_coord_y": UNITS_METER})
        t.add_units({"time": UNITS_SECOND})

        sizes = tuple(DATASET_SIZE_MAP.values())
        dataset = t.build_dataset("OBN Survey Receiver Gathers", sizes=sizes, header_dtype=structured_headers)

        assert dataset.metadata.name == "OBN Survey Receiver Gathers"
        assert dataset.metadata.attributes["surveyType"] == "3D"
        assert dataset.metadata.attributes["gatherType"] == "receiver_gathers"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers)

        # Verify seismic variable
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
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (1, 1, 512, 4096)
        assert seismic.metadata.stats_v1 is None
