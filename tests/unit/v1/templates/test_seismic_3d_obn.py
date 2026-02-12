"""Unit tests for Seismic3DObnReceiverGathersTemplate."""

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
from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)

# Typical OBN survey dimensions: 4 components, 500 receivers, 10 shot lines, 2 guns, 200 shot_index, 4096 samples
# Note: shot_index is a calculated dimension (0-N), shot_point is a coordinate
DATASET_SIZE_MAP = {"component": 4, "receiver": 500, "shot_line": 10, "gun": 2, "shot_index": 200, "time": 4096}
DATASET_DTYPE_MAP = {
    "component": "uint8",
    "receiver": "uint32",
    "shot_line": "uint32",
    "gun": "uint8",
    "time": "int32",
}
EXPECTED_COORDINATES = [
    "group_coord_x",
    "group_coord_y",
    "shot_point",
    "orig_field_record_num",
    "source_coord_x",
    "source_coord_y",
]


def _validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType, domain: str) -> None:
    """Validate the coordinate, headers, trace_mask variables in the dataset."""
    # Verify variables
    # 5 dim coords (excl. shot_index which is 0-N) + 6 non-dim coords + 1 data + 1 trace mask + 1 headers = 14 variables
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

    # Verify dimension coordinate variables (excluding shot_index which is calculated 0-N)
    for dim_name, dim_size in DATASET_SIZE_MAP.items():
        if dim_name == "shot_index":
            continue  # shot_index is calculated, no coordinate variable
        validate_variable(
            dataset,
            name=dim_name,
            dims=[(dim_name, dim_size)],
            coords=[dim_name],
            dtype=ScalarType(DATASET_DTYPE_MAP[dim_name]),
        )

    # Verify receiver coordinate variables (indexed by receiver only)
    for coord_name in ["group_coord_x", "group_coord_y"]:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=[("receiver", DATASET_SIZE_MAP["receiver"])],
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify source coordinate variables (indexed by shot_line, gun, shot_index)
    shot_dims = [(k, v) for k, v in DATASET_SIZE_MAP.items() if k in ["shot_line", "gun", "shot_index"]]
    for coord_name in ["source_coord_x", "source_coord_y"]:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=shot_dims,
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify shot_point coordinate (indexed by shot_line, gun, shot_index)
    validate_variable(
        dataset,
        name="shot_point",
        dims=shot_dims,
        coords=["shot_point"],
        dtype=ScalarType.UINT32,
    )

    # Verify orig_field_record_num coordinate
    validate_variable(
        dataset,
        name="orig_field_record_num",
        dims=shot_dims,
        coords=["orig_field_record_num"],
        dtype=ScalarType.UINT32,
    )


class TestSeismic3DObnReceiverGathersTemplate:
    """Unit tests for Seismic3DObnReceiverGathersTemplate."""

    def test_configuration(self) -> None:
        """Test template configuration and attributes."""
        t = Seismic3DObnReceiverGathersTemplate(data_domain="time")

        # Template attributes
        assert t.name == "ObnReceiverGathers3D"
        assert t._dim_names == ("component", "receiver", "shot_line", "gun", "shot_index", "time")
        assert t._calculated_dims == ("shot_index",)
        assert t._physical_coord_names == (
            "group_coord_x",
            "group_coord_y",
            "source_coord_x",
            "source_coord_y",
        )
        assert t._logical_coord_names == ("shot_point", "orig_field_record_num")
        assert t._var_chunk_shape == (1, 1, 1, 1, 512, 4096)

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == ()

        # Verify dataset attributes
        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyType": "3D", "gatherType": "common_receiver"}
        assert t.default_variable_name == "amplitude"

    def test_chunk_size_calculation(self) -> None:
        """Test that chunk shape produces approximately 6 MiB chunks.

        The chunk shape (1, 1, 1, 1, 512, 4096) produces:
        1 * 1 * 1 * 1 * 512 * 4096 = 2,097,152 samples.
        With float32 (4 bytes): 1,572,864 * 4 = 6,291,456 bytes = 6 MiB.
        """
        t = Seismic3DObnReceiverGathersTemplate(data_domain="time")

        # Get the chunk shape
        chunk_shape = t.full_chunk_shape
        assert chunk_shape == (1, 1, 1, 1, 512, 4096)

        # Calculate the number of samples per chunk
        samples_per_chunk = 1
        for dim_size in chunk_shape:
            samples_per_chunk *= dim_size

        # With float32 (4 bytes per sample), calculate chunk size in bytes
        bytes_per_chunk = samples_per_chunk * 4
        assert bytes_per_chunk == 8 * 1024 * 1024  # 8 MiB

    def test_chunking_optimized_for_shot_access(self) -> None:
        """Test that chunking is optimized for common-receiver gather access.

        The chunk shape (1, 1, 1, 1, 512, 4096) is designed for:
        - Single component per chunk
        - Single receiver per chunk (common-receiver gather access)
        - Single shot_line per chunk
        - All guns (typically 2-3) in one chunk
        - 512 shot_index values per chunk for efficient shot iteration
        """
        t = Seismic3DObnReceiverGathersTemplate(data_domain="time")

        # Dimensions: (component, receiver, shot_line, gun, shot_index, time)
        chunk_shape = t._var_chunk_shape
        assert chunk_shape[0] == 1  # component: single component per chunk
        assert chunk_shape[1] == 1  # receiver: single receiver per chunk
        assert chunk_shape[2] == 1  # shot_line: single shot line per chunk
        assert chunk_shape[3] == 1  # gun: all guns in one chunk (typical OBN has 2-3 guns)
        assert chunk_shape[4] == 512  # shot_index: good balance for shot iteration
        assert chunk_shape[5] == 4096  # time: full trace length

    def test_build_dataset(self, structured_headers: StructuredType) -> None:
        """Test building a complete dataset with the template."""
        t = Seismic3DObnReceiverGathersTemplate(data_domain="time")

        # Add units
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

        # Verify seismic amplitude variable
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
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (1, 1, 1, 1, 512, 4096)
        assert seismic.metadata.stats_v1 is None

    def test_depth_domain(self, structured_headers: StructuredType) -> None:
        """Test building a dataset with depth domain."""
        t = Seismic3DObnReceiverGathersTemplate(data_domain="depth")

        assert t.trace_domain == "depth"
        assert t._dim_names == ("component", "receiver", "shot_line", "gun", "shot_index", "depth")

        sizes = (4, 100, 5, 2, 50, 2048)  # Smaller sizes for this test
        dataset = t.build_dataset("ObnSurveyDepth", sizes=sizes, header_dtype=structured_headers)

        assert dataset.metadata.name == "ObnSurveyDepth"

        # Verify depth dimension coordinate exists
        depth_coord = next((v for v in dataset.variables if v.name == "depth"), None)
        assert depth_coord is not None
        assert depth_coord.dimensions[0].name == "depth"
        assert depth_coord.dimensions[0].size == 2048

    def test_calculated_dimension_names(self) -> None:
        """Test that calculated_dimension_names property returns shot_index."""
        t = Seismic3DObnReceiverGathersTemplate(data_domain="time")

        assert t.calculated_dimension_names == ("shot_index",)
