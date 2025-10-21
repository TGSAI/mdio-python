"""Unit tests for SeismicPreStackTemplate."""

import pytest
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
from mdio.builder.templates.seismic_3d_prestack_streamer_field_records import Seismic3DPreStackStreamerFieldRecordsTemplate

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)


def _validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType, domain: str) -> None:
    """Validate the coordinate, headers, trace_mask variables in the dataset."""
    # Verify variables
    # 6 dim coords + 5 non-dim coords + 1 data + 1 trace mask + 1 headers = 14 variables
    assert len(dataset.variables) == 14

    # Verify trace headers
    validate_variable(
        dataset,
        name="headers",
        dims=[("shot_line", 1), ("gun", 3), ("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["orig_field_record_num", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"],
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[("shot_line", 1), ("gun", 3), ("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["orig_field_record_num", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"],
        dtype=ScalarType.BOOL,
    )

    # Verify dimension coordinate variables
    for dim_name in ["shot_line", "gun", "shot_point", "cable", "channel", domain]:
        validate_variable(
            dataset,
            name=dim_name,
            dims=[
                (
                    dim_name,
                    {"shot_line": 1, "gun": 3, "shot_point": 256, "cable": 512, "channel": 24, domain: 2048}[dim_name],
                )
            ],
            coords=[dim_name],
            dtype=ScalarType.INT32,
        )

    # Verify non-dimension coordinate variables
    validate_variable(
        dataset,
        name="orig_field_record_num",
        dims=[("shot_line", 1), ("gun", 3), ("shot_point", 256)],
        coords=["orig_field_record_num"],
        dtype=ScalarType.INT32,
    )

    # Verify coordinate variables with units
    for coord_name in ["source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"]:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=[("shot_line", 1), ("gun", 3), ("shot_point", 256)]
            + ([("cable", 512), ("channel", 24)] if "group" in coord_name else []),
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER


class TestSeismic3DPreStackStreamerFieldRecordsTemplate:
    """Unit tests for Seismic3DPreStackStreamerFieldRecordsTemplate."""

    def test_configuration(self) -> None:
        """Unit tests for Seismic3DPreStackStreamerFieldRecordsTemplate."""
        t = Seismic3DPreStackStreamerFieldRecordsTemplate(data_domain="time")

        # Template attributes
        assert t.name == "PreStackStreamerFieldRecords3DTime"
        assert t._dim_names == ("shot_line", "gun", "shot_point", "cable", "channel", "time")
        assert t._physical_coord_names == ("source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y")
        # TODO(Anyone): Disable chunking in time domain when support is merged.
        # https://github.com/TGSAI/mdio-python/pull/723
        assert t.full_chunk_shape == (1, 1, 16, 1, 32, 1024)

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == ()

        # Verify dataset attributes
        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyDimensionality": "3D", "ensembleType": "shot_point", "processingStage": "pre-stack"}
        assert t.default_variable_name == "amplitude"

    def test_build_dataset(self, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DPreStackStreamerFieldRecordsTemplate build."""
        t = Seismic3DPreStackStreamerFieldRecordsTemplate(data_domain="time")
        t.add_units({"source_coord_x": UNITS_METER, "source_coord_y": UNITS_METER})  # spatial domain units
        t.add_units({"group_coord_x": UNITS_METER, "group_coord_y": UNITS_METER})  # spatial domain units
        t.add_units({"time": UNITS_SECOND})  # data domain units

        dataset = t.build_dataset(
            "North Sea 3D Streamer Field Records", sizes=(1, 3, 256, 512, 24, 2048), header_dtype=structured_headers
        )

        assert dataset.metadata.name == "North Sea 3D Streamer Field Records"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "shot_point"
        assert dataset.metadata.attributes["processingStage"] == "pre-stack"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, "time")

        # Verify seismic variable
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("shot_line", 1), ("gun", 3), ("shot_point", 256), ("cable", 512), ("channel", 24), ("time", 2048)],
            coords=["orig_field_record_num", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.cname == BloscCname.zstd
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (1, 1, 16, 1, 32, 1024)
        assert seismic.metadata.stats_v1 is None


@pytest.mark.parametrize("data_domain", ["Time", "TiME"])
def test_domain_case_handling(data_domain: str) -> None:
    """Test that domain parameter handles different cases correctly."""
    template = Seismic3DPreStackStreamerFieldRecordsTemplate(data_domain=data_domain)
    assert template._data_domain == data_domain.lower()
    assert template.name.endswith(data_domain.capitalize())
