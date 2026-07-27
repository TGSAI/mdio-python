"""Unit tests for Seismic3DCrgReceiverGathersTemplate."""

from tests.unit.v1.helpers import validate_variable

from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.schemas.v1.units import TimeUnitEnum
from mdio.builder.schemas.v1.units import TimeUnitModel
from mdio.builder.templates.seismic_3d_crg import Seismic3DCrgReceiverGathersTemplate

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)

# The per-receiver `trace` segment axis is inserted at ingest by HasDuplicates, so the
# template itself declares only the spatial receiver geometry plus the vertical axis.
DATASET_SIZE_MAP = {"component": 2, "receiver_line": 1, "receiver": 3, "time": 128}
DATASET_DTYPE_MAP = {
    "component": "uint8",
    "receiver_line": "uint32",
    "receiver": "uint32",
    "time": "int32",
}
EXPECTED_COORDINATES = ["group_coord_x", "group_coord_y"]
RECEIVER_DIMS = [("receiver_line", 1), ("receiver", 3)]


class TestSeismic3DCrgReceiverGathersTemplate:
    """Unit tests for Seismic3DCrgReceiverGathersTemplate."""

    def test_configuration(self) -> None:
        """Test template configuration and attributes."""
        t = Seismic3DCrgReceiverGathersTemplate(data_domain="time")

        assert t.name == "ObnContinuousReceiverGathers3D"
        assert t._dim_names == ("component", "receiver_line", "receiver", "time")
        assert t.spatial_dimension_names == ("component", "receiver_line", "receiver")
        # `trace` is inserted by HasDuplicates at ingest, not declared as a calculated dim.
        assert t._calculated_dims == ()
        assert t.synthesize_missing_dims == ("component",)
        assert t._physical_coord_names == ("group_coord_x", "group_coord_y")
        assert t._logical_coord_names == ()
        assert t._var_chunk_shape == (1, 1, 1, 16384)

        assert t._builder is None
        assert t._dim_sizes == ()

        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyType": "3D", "gatherType": "common_receiver"}
        assert t.default_variable_name == "amplitude"

    def test_whole_trace_chunk(self) -> None:
        """The vertical axis is chunked whole-trace (>= a real 15001-sample record)."""
        t = Seismic3DCrgReceiverGathersTemplate(data_domain="time")
        assert t.full_chunk_shape[-1] == 16384
        assert t.full_chunk_shape[-1] >= 15001

    def test_build_dataset(self, structured_headers: StructuredType) -> None:
        """Test building a complete dataset with the template."""
        t = Seismic3DCrgReceiverGathersTemplate(data_domain="time")
        t.add_units({"group_coord_x": UNITS_METER, "group_coord_y": UNITS_METER})
        t.add_units({"time": UNITS_SECOND})

        sizes = tuple(DATASET_SIZE_MAP.values())
        dataset = t.build_dataset("CrgSurvey3D", sizes=sizes, header_dtype=structured_headers)

        assert dataset.metadata.name == "CrgSurvey3D"
        assert dataset.metadata.attributes["gatherType"] == "common_receiver"
        assert dataset.metadata.attributes["defaultVariableName"] == "amplitude"

        # 4 dim coords + 2 non-dim coords + amplitude + trace_mask + headers = 9 variables.
        assert len(dataset.variables) == 9

        # Dimension coordinate variables.
        for dim_name, dim_size in DATASET_SIZE_MAP.items():
            validate_variable(
                dataset,
                name=dim_name,
                dims=[(dim_name, dim_size)],
                coords=[dim_name],
                dtype=ScalarType(DATASET_DTYPE_MAP[dim_name]),
            )

        # Receiver coordinate variables (indexed by receiver_line + receiver).
        for coord_name in EXPECTED_COORDINATES:
            coord = validate_variable(
                dataset,
                name=coord_name,
                dims=RECEIVER_DIMS,
                coords=[coord_name],
                dtype=ScalarType.FLOAT64,
            )
            assert coord.metadata.units_v1.length == LengthUnitEnum.METER

        # Headers and trace mask span the spatial dims only.
        validate_variable(
            dataset,
            name="headers",
            dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k != "time"],
            coords=EXPECTED_COORDINATES,
            dtype=structured_headers,
        )
        validate_variable(
            dataset,
            name="trace_mask",
            dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k != "time"],
            coords=EXPECTED_COORDINATES,
            dtype=ScalarType.BOOL,
        )

        # Seismic amplitude variable spans all declared dims and keeps the whole-trace chunk.
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
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (1, 1, 1, 16384)

    def test_depth_domain(self, structured_headers: StructuredType) -> None:
        """Test building a dataset with depth domain."""
        t = Seismic3DCrgReceiverGathersTemplate(data_domain="depth")

        assert t.trace_domain == "depth"
        assert t._dim_names == ("component", "receiver_line", "receiver", "depth")

        sizes = (1, 1, 2, 64)
        dataset = t.build_dataset("CrgSurveyDepth", sizes=sizes, header_dtype=structured_headers)

        depth_coord = next((v for v in dataset.variables if v.name == "depth"), None)
        assert depth_coord is not None
        assert depth_coord.dimensions[0].name == "depth"
        assert depth_coord.dimensions[0].size == 64


def _assert_coordinate_specs_match_build(template: Seismic3DCrgReceiverGathersTemplate) -> None:
    """declare_coordinate_specs must stay in sync with the built coordinates (base guard)."""
    specs = {spec.name: spec for spec in template.declare_coordinate_specs()}
    assert set(specs) == {"group_coord_x", "group_coord_y"}
    for spec in specs.values():
        assert spec.dimensions == ("receiver_line", "receiver")
        assert spec.dtype == ScalarType.FLOAT64


def test_declare_coordinate_specs() -> None:
    """The declared coordinate specs describe receiver X/Y over (receiver_line, receiver)."""
    _assert_coordinate_specs_match_build(Seismic3DCrgReceiverGathersTemplate())
