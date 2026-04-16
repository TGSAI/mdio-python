"""Unit tests for Seismic3DOffsetTilesTemplate."""

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
from mdio.builder.templates.seismic_3d_offset_tiles import Seismic3DOffsetTilesTemplate
from mdio.builder.templates.types import SeismicDataDomain

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
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
        dims=[("inline", 256), ("crossline", 256), ("inline_offset_tile", 12), ("crossline_offset_tile", 12)],
        coords=["cdp_x", "cdp_y"],
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[("inline", 256), ("crossline", 256), ("inline_offset_tile", 12), ("crossline_offset_tile", 12)],
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

    validate_variable(
        dataset,
        name="inline_offset_tile",
        dims=[("inline_offset_tile", 12)],
        coords=["inline_offset_tile"],
        dtype=ScalarType.INT16,
    )

    validate_variable(
        dataset,
        name="crossline_offset_tile",
        dims=[("crossline_offset_tile", 12)],
        coords=["crossline_offset_tile"],
        dtype=ScalarType.INT16,
    )

    domain_var = validate_variable(
        dataset,
        name=domain,
        dims=[(domain, 2048)],
        coords=[domain],
        dtype=ScalarType.INT32,
    )
    assert domain_var.metadata.units_v1 in (UNITS_METER, UNITS_SECOND)

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
class TestSeismic3DOffsetTilesTemplate:
    """Unit tests for Seismic3DOffsetTilesTemplate."""

    def test_configuration(self, data_domain: SeismicDataDomain) -> None:
        """Test template configuration and attributes."""
        t = Seismic3DOffsetTilesTemplate(data_domain=data_domain)

        # Template attributes
        assert t._dim_names == ("inline", "crossline", "inline_offset_tile", "crossline_offset_tile", data_domain)
        assert t._physical_coord_names == ("cdp_x", "cdp_y")
        assert t._logical_coord_names == ()
        assert t.full_chunk_shape == (4, 4, 6, 6, 4096)

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == ()

        # Verify dataset attributes
        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyType": "3D", "gatherType": "offset_tiles"}
        assert t.default_variable_name == "amplitude"

    def test_chunk_size_calculation(self, data_domain: SeismicDataDomain) -> None:
        """Test that chunk shape produces approximately 9 MiB chunks.

        The chunk shape (4, 4, 6, 6, 4096) produces:
        4 * 4 * 6 * 6 * 4096 = 2,359,296 samples.
        With float32 (4 bytes): 2,359,296 * 4 = 9,437,184 bytes = 9 MiB.
        """
        t = Seismic3DOffsetTilesTemplate(data_domain=data_domain)

        chunk_shape = t.full_chunk_shape
        assert chunk_shape == (4, 4, 6, 6, 4096)

        samples_per_chunk = 1
        for dim_size in chunk_shape:
            samples_per_chunk *= dim_size

        bytes_per_chunk = samples_per_chunk * 4
        assert bytes_per_chunk == 9 * 1024 * 1024  # 9 MiB

    def test_build_dataset(self, data_domain: SeismicDataDomain, structured_headers: StructuredType) -> None:
        """Test building a complete dataset with the template."""
        t = Seismic3DOffsetTilesTemplate(data_domain=data_domain)
        t.add_units({"cdp_x": UNITS_METER, "cdp_y": UNITS_METER})
        t.add_units({"time": UNITS_SECOND, "depth": UNITS_METER})

        dataset = t.build_dataset(
            "Offset Tile Gathers",
            sizes=(256, 256, 12, 12, 2048),
            header_dtype=structured_headers,
        )

        assert dataset.metadata.name == "Offset Tile Gathers"
        assert dataset.metadata.attributes["surveyType"] == "3D"
        assert dataset.metadata.attributes["gatherType"] == "offset_tiles"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, data_domain)

        # Verify seismic variable
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[
                ("inline", 256),
                ("crossline", 256),
                ("inline_offset_tile", 12),
                ("crossline_offset_tile", 12),
                (data_domain, 2048),
            ],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.cname == BloscCname.zstd
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == (4, 4, 6, 6, 4096)
        assert seismic.metadata.stats_v1 is None


@pytest.mark.parametrize("data_domain", ["Time", "DePTh"])
def test_domain_case_handling(data_domain: str) -> None:
    """Test that domain parameter handles different cases correctly."""
    template = Seismic3DOffsetTilesTemplate(data_domain=data_domain)
    assert template._data_domain == data_domain.lower()

    data_domain_suffix = data_domain.lower().capitalize()
    assert template.name == f"OffsetTiles3D{data_domain_suffix}"
