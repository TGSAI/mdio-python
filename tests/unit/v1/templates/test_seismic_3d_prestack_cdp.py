"""Unit tests for Seismic3DPreStackCDPTemplate."""

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
from mdio.builder.templates.seismic_3d_prestack_cdp import Seismic3DPreStackCDPTemplate
from mdio.builder.templates.types import CdpGatherDomain
from mdio.builder.templates.types import SeismicDataDomain

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_DEGREE = AngleUnitModel(angle=AngleUnitEnum.DEGREES)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)


def validate_coordinates_headers_trace_mask(
    dataset: Dataset,
    headers: StructuredType,
    data_domain: SeismicDataDomain,
    gather_domain: CdpGatherDomain,
) -> None:
    """A helper method to validate coordinates, headers, and trace mask."""
    # Verify variables
    # 4 dim coords + 2 non-dim coords + 1 data + 1 trace mask + 1 headers = 8 variables
    assert len(dataset.variables) == 9

    # Verify trace headers
    validate_variable(
        dataset,
        name="headers",
        dims=[("inline", 512), ("crossline", 768), (gather_domain, 36)],
        coords=["cdp_x", "cdp_y"],
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[("inline", 512), ("crossline", 768), (gather_domain, 36)],
        coords=["cdp_x", "cdp_y"],
        dtype=ScalarType.BOOL,
    )

    # Verify dimension coordinate variables
    validate_variable(
        dataset,
        name="inline",
        dims=[("inline", 512)],
        coords=["inline"],
        dtype=ScalarType.INT32,
    )

    validate_variable(
        dataset,
        name="crossline",
        dims=[("crossline", 768)],
        coords=["crossline"],
        dtype=ScalarType.INT32,
    )

    domain = validate_variable(
        dataset,
        name=gather_domain,
        dims=[(gather_domain, 36)],
        coords=[gather_domain],
        dtype=ScalarType.INT32,
    )
    assert domain.metadata.units_v1 in (UNITS_METER, UNITS_DEGREE)

    domain = validate_variable(
        dataset,
        name=data_domain,
        dims=[(data_domain, 1536)],
        coords=[data_domain],
        dtype=ScalarType.INT32,
    )
    assert domain.metadata.units_v1 in (UNITS_METER, UNITS_SECOND)

    # Verify non-dimension coordinate variables
    cdp_x = validate_variable(
        dataset,
        name="cdp_x",
        dims=[("inline", 512), ("crossline", 768), (gather_domain, 36)],
        coords=["cdp_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_x.metadata.units_v1 == UNITS_METER

    cdp_y = validate_variable(
        dataset,
        name="cdp_y",
        dims=[("inline", 512), ("crossline", 768), (gather_domain, 36)],
        coords=["cdp_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_y.metadata.units_v1 == UNITS_METER


@pytest.mark.parametrize("data_domain", ["depth", "time"])
@pytest.mark.parametrize("gather_domain", ["offset", "angle"])
class TestSeismic3DPreStackCDPTemplate:
    """Unit tests for Seismic3DPreStackCDPTemplate."""

    def test_configuration(self, data_domain: SeismicDataDomain, gather_domain: CdpGatherDomain) -> None:
        """Unit tests for Seismic3DPreStackCDPTemplate."""
        t = Seismic3DPreStackCDPTemplate(data_domain, gather_domain)

        # Template attributes for prestack CDP
        assert t._dim_names == ("inline", "crossline", gather_domain, data_domain)
        assert t._physical_coord_names == ("cdp_x", "cdp_y")
        assert t.full_chunk_shape == (8, 8, 32, 512)

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == ()

        # Verify prestack CDP attributes
        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyType": "3D", "gatherType": "cdp"}
        assert t.default_variable_name == "amplitude"

    def test_build_dataset(
        self,
        data_domain: SeismicDataDomain,
        gather_domain: CdpGatherDomain,
        structured_headers: StructuredType,
    ) -> None:
        """Unit tests for Seismic3DPreStackCDPDepthTemplate build."""
        t = Seismic3DPreStackCDPTemplate(data_domain, gather_domain)
        t.add_units({"cdp_x": UNITS_METER, "cdp_y": UNITS_METER})  # spatial domain units
        t.add_units({"offset": UNITS_METER, "angle": UNITS_DEGREE})  # gather domain units
        t.add_units({"time": UNITS_SECOND, "depth": UNITS_METER})  # data domain units

        gather_domain_suffix = gather_domain.capitalize()
        data_domain_suffix = data_domain.capitalize()
        assert t.name == f"PreStackCdp{gather_domain_suffix}Gathers3D{data_domain_suffix}"
        dataset = t.build_dataset("North Sea 3D Prestack", sizes=(512, 768, 36, 1536), header_dtype=structured_headers)

        assert dataset.metadata.name == "North Sea 3D Prestack"
        assert dataset.metadata.attributes["surveyType"] == "3D"
        assert dataset.metadata.attributes["gatherType"] == "cdp"

        validate_coordinates_headers_trace_mask(dataset, structured_headers, data_domain, gather_domain)

        # Verify seismic variable (prestack depth data)
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("inline", 512), ("crossline", 768), (gather_domain, 36), (data_domain, 1536)],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.cname == BloscCname.zstd
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (8, 8, 32, 512)
        assert seismic.metadata.stats_v1 is None


@pytest.mark.parametrize("data_domain", ["Time", "DePTh"])
@pytest.mark.parametrize("gather_domain", ["Offset", "OffSeT"])
def test_domain_case_handling(data_domain: str, gather_domain: str) -> None:
    """Test that domain parameter handles different cases correctly."""
    template = Seismic3DPreStackCDPTemplate(data_domain, gather_domain)
    assert template._data_domain == data_domain.lower()

    gather_domain_suffix = gather_domain.lower().capitalize()
    data_domain_suffix = data_domain.lower().capitalize()
    assert template.name == f"PreStackCdp{gather_domain_suffix}Gathers3D{data_domain_suffix}"
