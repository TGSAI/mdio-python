"""Unit tests for Seismic3DPreStackShotTemplate."""

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
from mdio.builder.templates.seismic_3d_prestack_shot import Seismic3DPreStackShotTemplate

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)


def _validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType, domain: str) -> None:
    """Validate the coordinate, headers, trace_mask variables in the dataset."""
    # Verify variables
    # 4 dim coords + 5 non-dim coords + 1 data + 1 trace mask + 1 headers = 12 variables
    assert len(dataset.variables) == 12

    # Verify trace headers
    validate_variable(
        dataset,
        name="headers",
        dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["gun", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"],
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["gun", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"],
        dtype=ScalarType.BOOL,
    )

    # Verify dimension coordinate variables
    validate_variable(
        dataset,
        name="shot_point",
        dims=[("shot_point", 256)],
        coords=["shot_point"],
        dtype=ScalarType.INT32,
    )

    validate_variable(
        dataset,
        name="cable",
        dims=[("cable", 512)],
        coords=["cable"],
        dtype=ScalarType.INT32,
    )

    validate_variable(
        dataset,
        name="channel",
        dims=[("channel", 24)],
        coords=["channel"],
        dtype=ScalarType.INT32,
    )

    domain = validate_variable(
        dataset,
        name=domain,
        dims=[(domain, 2048)],
        coords=[domain],
        dtype=ScalarType.INT32,
    )
    assert domain.metadata.units_v1 == UNITS_SECOND

    # Verify non-dimension coordinate variables
    validate_variable(
        dataset,
        name="gun",
        dims=[("shot_point", 256)],
        coords=["gun"],
        dtype=ScalarType.UINT8,
    )

    source_coord_x = validate_variable(
        dataset,
        name="source_coord_x",
        dims=[("shot_point", 256)],
        coords=["source_coord_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert source_coord_x.metadata.units_v1 == UNITS_METER

    source_coord_y = validate_variable(
        dataset,
        name="source_coord_y",
        dims=[("shot_point", 256)],
        coords=["source_coord_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert source_coord_y.metadata.units_v1 == UNITS_METER

    group_coord_x = validate_variable(
        dataset,
        name="group_coord_x",
        dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["group_coord_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert group_coord_x.metadata.units_v1 == UNITS_METER

    group_coord_y = validate_variable(
        dataset,
        name="group_coord_y",
        dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["group_coord_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert group_coord_y.metadata.units_v1 == UNITS_METER


class TestSeismic3DPreStackShotTemplate:
    """Unit tests for Seismic3DPreStackShotTemplate."""

    def test_configuration(self) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate in time domain."""
        t = Seismic3DPreStackShotTemplate(data_domain="time")

        # Template attributes for prestack shot
        assert t._data_domain == "time"
        assert t._dim_names == ("shot_point", "cable", "channel", "time")
        assert t._physical_coord_names == ("source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y")
        assert t._logical_coord_names == ("gun",)
        assert t.full_chunk_shape == (8, 1, 128, 2048)

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == ()

        # Verify prestack shot attributes
        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyType": "3D", "ensembleType": "common_source"}

        assert t.name == "PreStackShotGathers3DTime"

    def test_build_dataset(self, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate build in time domain."""
        t = Seismic3DPreStackShotTemplate(data_domain="time")
        t.add_units({"source_coord_x": UNITS_METER, "source_coord_y": UNITS_METER})  # spatial domain units
        t.add_units({"group_coord_x": UNITS_METER, "group_coord_y": UNITS_METER})  # spatial domain units
        t.add_units({"time": UNITS_SECOND})  # data domain units

        assert t.name == "PreStackShotGathers3DTime"
        dataset = t.build_dataset("North Sea 3D Shot Time", sizes=(256, 512, 24, 2048), header_dtype=structured_headers)

        assert dataset.metadata.name == "North Sea 3D Shot Time"
        assert dataset.metadata.attributes["surveyType"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "common_source"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, "time")

        # Verify seismic variable (prestack shot time data)
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("shot_point", 256), ("cable", 512), ("channel", 24), ("time", 2048)],
            coords=["gun", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.cname == BloscCname.zstd
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (8, 1, 128, 2048)
        assert seismic.metadata.stats_v1 is None


@pytest.mark.parametrize("data_domain", ["Time", "TiME"])
def test_domain_case_handling(data_domain: str) -> None:
    """Test that domain parameter handles different cases correctly."""
    template = Seismic3DPreStackShotTemplate(data_domain=data_domain)
    assert template._data_domain == data_domain.lower()
    assert template.name.endswith(data_domain.capitalize())
