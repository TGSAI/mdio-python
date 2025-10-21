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
from mdio.builder.templates.seismic_prestack import SeismicPreStackTemplate

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
    shot_line = validate_variable(
        dataset,
        name="shot_line",
        dims=[("shot_line", 1)],
        coords=["shot_line"],
        dtype=ScalarType.INT32,
    )
    assert shot_line.metadata is None

    gun = validate_variable(
        dataset,
        name="gun",
        dims=[("gun", 3)],
        coords=["gun"],
        dtype=ScalarType.INT32,
    )
    assert gun.metadata is None

    shot_point = validate_variable(
        dataset,
        name="shot_point",
        dims=[("shot_point", 256)],
        coords=["shot_point"],
        dtype=ScalarType.INT32,
    )
    assert shot_point.metadata is None

    cable = validate_variable(
        dataset,
        name="cable",
        dims=[("cable", 512)],
        coords=["cable"],
        dtype=ScalarType.INT32,
    )
    assert cable.metadata is None

    channel = validate_variable(
        dataset,
        name="channel",
        dims=[("channel", 24)],
        coords=["channel"],
        dtype=ScalarType.INT32,
    )
    assert channel.metadata is None

    domain_var = validate_variable(
        dataset,
        name=domain,
        dims=[(domain, 2048)],
        coords=[domain],
        dtype=ScalarType.INT32,
    )
    assert domain_var.metadata is None

    # Verify non-dimension coordinate variables
    validate_variable(
        dataset,
        name="orig_field_record_num",
        dims=[("shot_line", 1), ("gun", 3), ("shot_point", 256)],
        coords=["orig_field_record_num"],
        dtype=ScalarType.INT32,
    )

    source_coord_x = validate_variable(
        dataset,
        name="source_coord_x",
        dims=[("shot_line", 1), ("gun", 3), ("shot_point", 256)],
        coords=["source_coord_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert source_coord_x.metadata.units_v1.length == LengthUnitEnum.METER

    source_coord_y = validate_variable(
        dataset,
        name="source_coord_y",
        dims=[("shot_line", 1), ("gun", 3), ("shot_point", 256)],
        coords=["source_coord_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert source_coord_y.metadata.units_v1.length == LengthUnitEnum.METER

    group_coord_x = validate_variable(
        dataset,
        name="group_coord_x",
        dims=[("shot_line", 1), ("gun", 3), ("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["group_coord_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert group_coord_x.metadata.units_v1.length == LengthUnitEnum.METER

    group_coord_y = validate_variable(
        dataset,
        name="group_coord_y",
        dims=[("shot_line", 1), ("gun", 3), ("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["group_coord_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert group_coord_y.metadata.units_v1.length == LengthUnitEnum.METER


class TestSeismic3DPreStackShotTemplate:
    """Unit tests for SeismicPreStackTemplate."""

    def test_configuration(self) -> None:
        """Unit tests for SeismicPreStackTemplate in time domain."""
        t = SeismicPreStackTemplate(data_domain="time")

        # Template attributes for prestack shot
        assert t.name == "PreStackGathers3DTime"
        assert t.default_variable_name == "amplitude"
        assert t.trace_domain == "time"
        assert t.spatial_dimension_names == ("shot_line", "gun", "shot_point", "cable", "channel")
        assert t.dimension_names == ("shot_line", "gun", "shot_point", "cable", "channel", "time")
        assert t.physical_coordinate_names == ("source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y")
        assert t.logical_coordinate_names == ("orig_field_record_num",)
        assert t.coordinate_names == (
            "source_coord_x",
            "source_coord_y",
            "group_coord_x",
            "group_coord_y",
            "orig_field_record_num",
        )
        assert t.full_chunk_shape == (1, 1, 16, 1, 32, 1024)

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == ()
        assert t._units == {}

        # Verify prestack shot attributes
        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyDimensionality": "3D", "ensembleType": "shot_point", "processingStage": "pre-stack"}
        assert t.default_variable_name == "amplitude"

        assert t.name == "PreStackGathers3DTime"

    def test_build_dataset(self, structured_headers: StructuredType) -> None:
        """Unit tests for SeismicPreStackTemplate build in time domain."""
        t = SeismicPreStackTemplate(data_domain="time")
        t.add_units({"source_coord_x": UNITS_METER, "source_coord_y": UNITS_METER})  # spatial domain units
        t.add_units({"group_coord_x": UNITS_METER, "group_coord_y": UNITS_METER})  # spatial domain units
        t.add_units({"time": UNITS_SECOND})  # data domain units

        dataset = t.build_dataset(
            "North Sea 3D Shot Time", sizes=(1, 3, 256, 512, 24, 2048), header_dtype=structured_headers
        )

        assert dataset.metadata.name == "North Sea 3D Shot Time"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "shot_point"
        assert dataset.metadata.attributes["processingStage"] == "pre-stack"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, "time")

        # Verify seismic variable (prestack shot time data)
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
    template = SeismicPreStackTemplate(data_domain=data_domain)
    assert template._data_domain == data_domain.lower()
    assert template.name.endswith(data_domain.capitalize())
