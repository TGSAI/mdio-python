"""Unit tests for Seismic3DPreStackCocaTemplate."""

import pytest
from tests.unit.v1.helpers import validate_variable

from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.v1.dataset import Dataset
from mdio.builder.schemas.v1.units import AngleUnitEnum
from mdio.builder.schemas.v1.units import AngleUnitModel
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.schemas.v1.units import TimeUnitEnum
from mdio.builder.schemas.v1.units import TimeUnitModel
from mdio.builder.templates.seismic_3d_prestack_coca import Seismic3DPreStackCocaTemplate
from mdio.builder.templates.types import SeismicDataDomain

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_DEGREE = AngleUnitModel(angle=AngleUnitEnum.DEGREES)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)


def _validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType, domain: str) -> None:
    """Validate the coordinate, headers, trace_mask variables in the dataset."""
    # Verify variables
    # 5 dim coords + 2 non-dim coords + 1 data + 1 trace mask + 1 headers = 10 variables
    assert len(dataset.variables) == 10

    # Verify trace headers
    validate_variable(
        dataset,
        name="headers",
        dims=[("inline", 256), ("crossline", 256), ("offset", 100), ("azimuth", 6)],
        coords=["cdp_x", "cdp_y"],
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[("inline", 256), ("crossline", 256), ("offset", 100), ("azimuth", 6)],
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
        dims=[("crossline", 256)],
        coords=["crossline"],
        dtype=ScalarType.INT32,
    )

    offset = validate_variable(
        dataset,
        name="offset",
        dims=[("offset", 100)],
        coords=["offset"],
        dtype=ScalarType.INT32,
    )
    assert offset.metadata.units_v1 == UNITS_METER

    azimuth = validate_variable(
        dataset,
        name="azimuth",
        dims=[("azimuth", 6)],
        coords=["azimuth"],
        dtype=ScalarType.FLOAT32,
    )
    assert azimuth.metadata.units_v1 == UNITS_DEGREE

    domain = validate_variable(
        dataset,
        name=domain,
        dims=[(domain, 2048)],
        coords=[domain],
        dtype=ScalarType.INT32,
    )
    assert domain.metadata.units_v1 in (UNITS_METER, UNITS_SECOND)

    # Verify non-dimension coordinate variables
    cdp_x = validate_variable(
        dataset,
        name="cdp_x",
        dims=[("inline", 256), ("crossline", 256)],
        coords=["cdp_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_x.metadata.units_v1 == UNITS_METER

    cdp_y = validate_variable(
        dataset,
        name="cdp_y",
        dims=[("inline", 256), ("crossline", 256)],
        coords=["cdp_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_y.metadata.units_v1 == UNITS_METER


@pytest.mark.parametrize("data_domain", ["depth", "time"])
class TestSeismic3DPreStackCocaTemplate:
    """Unit tests for Seismic3DPreStackCocaTemplate."""

    def test_configuration(self, data_domain: SeismicDataDomain) -> None:
        """Unit tests for Seismic3DPreStackCocaTemplate."""
        t = Seismic3DPreStackCocaTemplate(data_domain=data_domain)

        # Template attributes
        assert t._dim_names == ("inline", "crossline", "offset", "azimuth", data_domain)
        assert t._physical_coord_names == ("cdp_x", "cdp_y")
        assert t.full_chunk_shape == (8, 8, 32, 1, 1024)

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == ()

        # Verify dataset attributes
        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyType": "3D", "gatherType": "common_offset_common_azimuth"}
        assert t.default_variable_name == "amplitude"

    def test_build_dataset(self, data_domain: SeismicDataDomain, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate build."""
        t = Seismic3DPreStackCocaTemplate(data_domain=data_domain)
        t.add_units({"cdp_x": UNITS_METER, "cdp_y": UNITS_METER})  # spatial domain units
        t.add_units({"offset": UNITS_METER, "azimuth": UNITS_DEGREE})  # spatial domain units
        t.add_units({"time": UNITS_SECOND, "depth": UNITS_METER})  # data domain units

        dataset = t.build_dataset(
            "Permian Basin 3D CDP Coca Gathers", sizes=(256, 256, 100, 6, 2048), header_dtype=structured_headers
        )

        assert dataset.metadata.name == "Permian Basin 3D CDP Coca Gathers"
        assert dataset.metadata.attributes["surveyType"] == "3D"
        assert dataset.metadata.attributes["gatherType"] == "common_offset_common_azimuth"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, data_domain)

        # Verify seismic variable (prestack shot depth data)
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("inline", 256), ("crossline", 256), ("offset", 100), ("azimuth", 6), (data_domain, 2048)],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.cname == BloscCname.zstd
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (8, 8, 32, 1, 1024)
        assert seismic.metadata.stats_v1 is None


@pytest.mark.parametrize("data_domain", ["Time", "DePTh"])
def test_domain_case_handling(data_domain: str) -> None:
    """Test that domain parameter handles different cases correctly."""
    template = Seismic3DPreStackCocaTemplate(data_domain=data_domain)
    assert template._data_domain == data_domain.lower()

    data_domain_suffix = data_domain.lower().capitalize()
    assert template.name == f"PreStackCocaGathers3D{data_domain_suffix}"
