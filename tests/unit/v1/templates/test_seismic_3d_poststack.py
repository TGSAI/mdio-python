"""Unit tests for Seismic3DPostStackTemplate."""

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
from mdio.builder.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
from mdio.builder.templates.types import SeismicDataDomain

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)


def _validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType, domain: str) -> None:
    """Validate the coordinate, headers, trace_mask variables in the dataset."""
    # Verify variables
    # 3 dim coords + 2 non-dim coords + 1 data + 1 trace mask + 1 headers = 7 variables
    assert len(dataset.variables) == 8

    # Verify trace headers
    validate_variable(
        dataset,
        name="headers",
        dims=[("inline", 256), ("crossline", 512)],
        coords=["cdp_x", "cdp_y"],
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[("inline", 256), ("crossline", 512)],
        coords=["cdp_x", "cdp_y"],
        dtype=ScalarType.BOOL,
    )

    # Verify dimension coordinate variables
    validate_variable(
        dataset,
        name="inline",
        dims=[("inline", 256)],
        coords=["inline"],
        dtype=ScalarType.INT32,
    )

    validate_variable(
        dataset,
        name="crossline",
        dims=[("crossline", 512)],
        coords=["crossline"],
        dtype=ScalarType.INT32,
    )

    domain = validate_variable(
        dataset,
        name=domain,
        dims=[(domain, 1024)],
        coords=[domain],
        dtype=ScalarType.INT32,
    )
    assert domain.metadata.units_v1 in (UNITS_METER, UNITS_SECOND)

    # Verify non-dimension coordinate variables
    cdp_x = validate_variable(
        dataset,
        name="cdp_x",
        dims=[("inline", 256), ("crossline", 512)],
        coords=["cdp_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_x.metadata.units_v1 == UNITS_METER

    cdp_y = validate_variable(
        dataset,
        name="cdp_y",
        dims=[("inline", 256), ("crossline", 512)],
        coords=["cdp_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_y.metadata.units_v1 == UNITS_METER


@pytest.mark.parametrize("data_domain", ["depth", "time"])
class TestSeismic3DPostStackTemplate:
    """Unit tests for Seismic3DPostStackTemplate."""

    def test_configuration(self, data_domain: SeismicDataDomain) -> None:
        """Unit tests for Seismic3DPostStackTemplate."""
        t = Seismic3DPostStackTemplate(data_domain=data_domain)

        # Template attributes to be overridden by subclasses
        assert t._data_domain == data_domain  # Domain should be lowercased
        assert t._dim_names == ("inline", "crossline", data_domain)
        assert t._physical_coord_names == ("cdp_x", "cdp_y")
        assert t.full_chunk_shape == (128, 128, 128)

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == ()

        # Verify dataset attributes
        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyType": "3D", "gatherType": "stacked"}
        assert t.default_variable_name == "amplitude"

    def test_build_dataset(self, data_domain: SeismicDataDomain, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DPostStackTemplate build."""
        t = Seismic3DPostStackTemplate(data_domain=data_domain)
        t.add_units({"cdp_x": UNITS_METER, "cdp_y": UNITS_METER})  # spatial domain units
        t.add_units({"time": UNITS_SECOND, "depth": UNITS_METER})  # data domain units

        data_domain_suffix = data_domain.capitalize()
        assert t.name == f"PostStack3D{data_domain_suffix}"
        dataset = t.build_dataset("Seismic 3D", sizes=(256, 512, 1024), header_dtype=structured_headers)

        assert dataset.metadata.name == "Seismic 3D"
        assert dataset.metadata.attributes["surveyType"] == "3D"
        assert dataset.metadata.attributes["gatherType"] == "stacked"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, data_domain)

        # Verify seismic variable
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("inline", 256), ("crossline", 512), (data_domain, 1024)],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.cname == BloscCname.zstd
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (128, 128, 128)
        assert seismic.metadata.stats_v1 is None


@pytest.mark.parametrize("data_domain", ["Time", "DePTh"])
def test_domain_case_handling(data_domain: str) -> None:
    """Test that domain parameter handles different cases correctly."""
    template = Seismic3DPostStackTemplate(data_domain=data_domain)
    assert template._data_domain == data_domain.lower()

    data_domain_suffix = data_domain.lower().capitalize()
    assert template.name == f"PostStack3D{data_domain_suffix}"
