"""Unit tests for Seismic3DPreStackCDPTemplate."""

from tests.unit.v1.helpers import validate_variable

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.templates.seismic_3d_prestack_cdp import Seismic3DPreStackCDPTemplate
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.units import TimeUnitEnum
from mdio.schemas.v1.units import TimeUnitModel

_UNIT_METER = AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))
_UNIT_SECOND = AllUnits(units_v1=TimeUnitModel(time=TimeUnitEnum.SECOND))


def validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType) -> None:
    """A helper method to validate coordinates, headers, and trace mask."""
    # Verify variables
    # 3 dim coords + 2 non-dim coords + 1 data + 1 trace mask + 1 headers = 8 variables
    assert len(dataset.variables) == 8

    # Verify trace headers
    validate_variable(
        dataset,
        name="headers",
        dims=[("inline", 512), ("crossline", 768), ("offset", 36)],
        coords=["cdp_x", "cdp_y"],
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[("inline", 512), ("crossline", 768), ("offset", 36)],
        coords=["cdp_x", "cdp_y"],
        dtype=ScalarType.BOOL,
    )

    # Verify dimension coordinate variables
    inline = validate_variable(
        dataset,
        name="inline",
        dims=[("inline", 512)],
        coords=["inline"],
        dtype=ScalarType.INT32,
    )
    assert inline.metadata is None

    crossline = validate_variable(
        dataset,
        name="crossline",
        dims=[("crossline", 768)],
        coords=["crossline"],
        dtype=ScalarType.INT32,
    )
    assert crossline.metadata is None

    crossline = validate_variable(
        dataset,
        name="offset",
        dims=[("offset", 36)],
        coords=["offset"],
        dtype=ScalarType.INT32,
    )
    assert crossline.metadata is None

    # Verify non-dimension coordinate variables
    cdp_x = validate_variable(
        dataset,
        name="cdp_x",
        dims=[("inline", 512), ("crossline", 768), ("offset", 36)],
        coords=["cdp_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_x.metadata.units_v1.length == LengthUnitEnum.METER

    cdp_y = validate_variable(
        dataset,
        name="cdp_y",
        dims=[("inline", 512), ("crossline", 768), ("offset", 36)],
        coords=["cdp_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_y.metadata.units_v1.length == LengthUnitEnum.METER


class TestSeismic3DPreStackCDPTemplate:
    """Unit tests for Seismic3DPreStackCDPTemplate."""

    def test_configuration_depth(self) -> None:
        """Unit tests for Seismic3DPreStackCDPTemplate."""
        t = Seismic3DPreStackCDPTemplate(domain="DEPTH")

        # Template attributes for prestack CDP
        assert t._trace_domain == "depth"
        assert t._coord_dim_names == ["inline", "crossline", "offset"]
        assert t._dim_names == ["inline", "crossline", "offset", "depth"]
        assert t._coord_names == ["cdp_x", "cdp_y"]
        assert t._var_chunk_shape == [1, 1, 512, 4096]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._coord_units == []

        # Verify prestack CDP attributes
        attrs = t._load_dataset_attributes()
        assert attrs.attributes == {
            "surveyDimensionality": "3D",
            "ensembleType": "cdp",
            "processingStage": "pre-stack",
        }
        assert t.trace_variable_name == "amplitude"

    def test_configuration_time(self) -> None:
        """Unit tests for Seismic3DPreStackCDPTemplate."""
        t = Seismic3DPreStackCDPTemplate(domain="TIME")

        # Template attributes for prestack CDP
        assert t._trace_domain == "time"
        assert t._coord_dim_names == ["inline", "crossline", "offset"]
        assert t._dim_names == ["inline", "crossline", "offset", "time"]
        assert t._coord_names == ["cdp_x", "cdp_y"]
        assert t._var_chunk_shape == [1, 1, 512, 4096]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._coord_units == []

        # Verify prestack CDP attributes
        attrs = t._load_dataset_attributes()
        assert attrs.attributes == {
            "surveyDimensionality": "3D",
            "ensembleType": "cdp",
            "processingStage": "pre-stack",
        }

        assert t.name == "PreStackCdpGathers3DTime"

    def test_domain_case_handling(self) -> None:
        """Test that domain parameter handles different cases correctly."""
        # Test uppercase
        t1 = Seismic3DPreStackCDPTemplate("ELEVATION")
        assert t1._trace_domain == "elevation"
        assert t1.name == "PreStackCdpGathers3DElevation"

        # Test mixed case
        t2 = Seismic3DPreStackCDPTemplate("elevatioN")
        assert t2._trace_domain == "elevation"
        assert t2.name == "PreStackCdpGathers3DElevation"

    def test_build_dataset_depth(self, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DPreStackCDPDepthTemplate build with depth domain."""
        t = Seismic3DPreStackCDPTemplate(domain="depth")

        assert t.name == "PreStackCdpGathers3DDepth"
        dataset = t.build_dataset(
            "North Sea 3D Prestack Depth",
            sizes=[512, 768, 36, 1536],
            coord_units=[_UNIT_METER, _UNIT_METER],
            headers=structured_headers,
        )

        assert dataset.metadata.name == "North Sea 3D Prestack Depth"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "cdp"
        assert dataset.metadata.attributes["processingStage"] == "pre-stack"

        validate_coordinates_headers_trace_mask(dataset, structured_headers)

        # Verify seismic variable (prestack depth data)
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("inline", 512), ("crossline", 768), ("offset", 36), ("depth", 1536)],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.algorithm == "zstd"
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [1, 1, 512, 4096]
        assert seismic.metadata.stats_v1 is None

    def test_build_dataset_time(self, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DPreStackCDPTimeTemplate build with time domain."""
        t = Seismic3DPreStackCDPTemplate(domain="time")

        assert t.name == "PreStackCdpGathers3DTime"
        dataset = t.build_dataset(
            "Santos Basin 3D Prestack",
            sizes=[512, 768, 36, 1536],
            coord_units=[_UNIT_METER, _UNIT_METER],
            headers=structured_headers,
        )

        assert dataset.metadata.name == "Santos Basin 3D Prestack"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "cdp"
        assert dataset.metadata.attributes["processingStage"] == "pre-stack"

        validate_coordinates_headers_trace_mask(dataset, structured_headers)

        # Verify seismic variable (prestack time data)
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("inline", 512), ("crossline", 768), ("offset", 36), ("time", 1536)],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.algorithm == "zstd"
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [1, 1, 512, 4096]
        assert seismic.metadata.stats_v1 is None
