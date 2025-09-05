"""Unit tests for Seismic3DPreStackCocaTemplate."""

from tests.unit.v1.helpers import validate_variable

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.compressors import Blosc
from mdio.schemas.compressors import BloscCname
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.templates.seismic_3d_prestack_coca import Seismic3DPreStackCocaTemplate
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import AngleUnitEnum
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.units import TimeUnitEnum
from mdio.schemas.v1.units import TimeUnitModel

_UNIT_METER = AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))
_UNIT_SECOND = AllUnits(units_v1=TimeUnitModel(time=TimeUnitEnum.SECOND))


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
    inline = validate_variable(
        dataset,
        name="inline",
        dims=[("inline", 256)],
        coords=["inline"],
        dtype=ScalarType.INT32,
    )
    assert inline.metadata is None

    crossline = validate_variable(
        dataset,
        name="crossline",
        dims=[("crossline", 256)],
        coords=["crossline"],
        dtype=ScalarType.INT32,
    )
    assert crossline.metadata is None

    offset = validate_variable(
        dataset,
        name="offset",
        dims=[("offset", 100)],
        coords=["offset"],
        dtype=ScalarType.INT32,
    )
    assert offset.metadata.units_v1.length == LengthUnitEnum.METER

    azimuth = validate_variable(
        dataset,
        name="azimuth",
        dims=[("azimuth", 6)],
        coords=["azimuth"],
        dtype=ScalarType.FLOAT32,
    )
    assert azimuth.metadata.units_v1.angle == AngleUnitEnum.DEGREES

    domain = validate_variable(
        dataset,
        name=domain,
        dims=[(domain, 2048)],
        coords=[domain],
        dtype=ScalarType.INT32,
    )
    assert domain.metadata is None

    # Verify non-dimension coordinate variables
    cdp_x = validate_variable(
        dataset,
        name="cdp_x",
        dims=[("inline", 256), ("crossline", 256)],
        coords=["cdp_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_x.metadata.units_v1.length == LengthUnitEnum.METER

    cdp_y = validate_variable(
        dataset,
        name="cdp_y",
        dims=[("inline", 256), ("crossline", 256)],
        coords=["cdp_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_y.metadata.units_v1.length == LengthUnitEnum.METER


class TestSeismic3DPreStackCocaTemplate:
    """Unit tests for Seismic3DPreStackCocaTemplate."""

    def test_configuration_time(self) -> None:
        """Unit tests for Seismic3DPreStackCocaTemplate in time domain."""
        t = Seismic3DPreStackCocaTemplate(domain="time")

        # Template attributes
        assert t._coord_dim_names == ["inline", "crossline", "offset", "azimuth"]
        assert t._dim_names == ["inline", "crossline", "offset", "azimuth", "time"]
        assert t._coord_names == ["cdp_x", "cdp_y"]
        assert t._var_chunk_shape == [8, 8, 32, 1, 1024]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._horizontal_coord_unit is None

        # Verify dataset attributes
        attrs = t._load_dataset_attributes()
        assert attrs.attributes == {
            "surveyDimensionality": "3D",
            "ensembleType": "cdp_coca",
            "processingStage": "pre-stack",
        }
        assert t.default_variable_name == "amplitude"

    def test_build_dataset_time(self, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate build in time domain."""
        t = Seismic3DPreStackCocaTemplate(domain="time")

        dataset = t.build_dataset(
            "Permian Basin 3D CDP Coca Gathers",
            sizes=[256, 256, 100, 6, 2048],
            horizontal_coord_unit=_UNIT_METER,
            headers=structured_headers,
        )

        assert dataset.metadata.name == "Permian Basin 3D CDP Coca Gathers"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "cdp_coca"
        assert dataset.metadata.attributes["processingStage"] == "pre-stack"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, "time")

        # Verify seismic variable (prestack shot depth data)
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("inline", 256), ("crossline", 256), ("offset", 100), ("azimuth", 6), ("time", 2048)],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.cname == BloscCname.zstd
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [8, 8, 32, 1, 1024]
        assert seismic.metadata.stats_v1 is None
