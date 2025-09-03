"""Unit tests for Seismic3DPostStackTemplate."""

from tests.unit.v1.helpers import validate_variable

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.units import TimeUnitEnum
from mdio.schemas.v1.units import TimeUnitModel

_UNIT_METER = AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))
_UNIT_SECOND = AllUnits(units_v1=TimeUnitModel(time=TimeUnitEnum.SECOND))


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
        dims=[("crossline", 512)],
        coords=["crossline"],
        dtype=ScalarType.INT32,
    )
    assert crossline.metadata is None

    domain = validate_variable(
        dataset,
        name=domain,
        dims=[(domain, 1024)],
        coords=[domain],
        dtype=ScalarType.INT32,
    )
    assert domain.metadata is None

    # Verify non-dimension coordinate variables
    cdp_x = validate_variable(
        dataset,
        name="cdp_x",
        dims=[("inline", 256), ("crossline", 512)],
        coords=["cdp_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_x.metadata.units_v1.length == LengthUnitEnum.METER

    cdp_y = validate_variable(
        dataset,
        name="cdp_y",
        dims=[("inline", 256), ("crossline", 512)],
        coords=["cdp_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_y.metadata.units_v1.length == LengthUnitEnum.METER


class TestSeismic3DPostStackTemplate:
    """Unit tests for Seismic3DPostStackTemplate."""

    def test_configuration_depth(self) -> None:
        """Unit tests for Seismic3DPostStackTemplate with depth domain."""
        t = Seismic3DPostStackTemplate(domain="depth")

        # Template attributes to be overridden by subclasses
        assert t._trace_domain == "depth"  # Domain should be lowercased
        assert t._coord_dim_names == ["inline", "crossline"]
        assert t._dim_names == ["inline", "crossline", "depth"]
        assert t._coord_names == ["cdp_x", "cdp_y"]
        assert t._var_chunk_shape == [128, 128, 128]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._horizontal_coord_unit is None

        # Verify dataset attributes
        attrs = t._load_dataset_attributes()
        assert attrs.attributes == {
            "surveyDimensionality": "3D",
            "ensembleType": "line",
            "processingStage": "post-stack",
        }
        assert t.default_variable_name == "amplitude"

    def test_configuration_time(self) -> None:
        """Unit tests for Seismic3DPostStackTemplate with time domain."""
        t = Seismic3DPostStackTemplate(domain="time")

        # Template attributes to be overridden by subclasses
        assert t._trace_domain == "time"  # Domain should be lowercased
        assert t._coord_dim_names == ["inline", "crossline"]
        assert t._dim_names == ["inline", "crossline", "time"]
        assert t._coord_names == ["cdp_x", "cdp_y"]
        assert t._var_chunk_shape == [128, 128, 128]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._horizontal_coord_unit is None

        assert t._load_dataset_attributes().attributes == {
            "surveyDimensionality": "3D",
            "ensembleType": "line",
            "processingStage": "post-stack",
        }

        assert t.name == "PostStack3DTime"

    def test_domain_case_handling(self) -> None:
        """Test that domain parameter handles different cases correctly."""
        # Test uppercase
        t1 = Seismic3DPostStackTemplate("ELEVATION")
        assert t1._trace_domain == "elevation"
        assert t1.name == "PostStack3DElevation"

        # Test mixed case
        t2 = Seismic3DPostStackTemplate("elevatioN")
        assert t2._trace_domain == "elevation"
        assert t2.name == "PostStack3DElevation"

    def test_build_dataset_depth(self, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DPostStackTemplate build with depth domain."""
        t = Seismic3DPostStackTemplate(domain="depth")

        assert t.name == "PostStack3DDepth"
        dataset = t.build_dataset(
            "Seismic 3D",
            sizes=[256, 512, 1024],
            horizontal_coord_unit=_UNIT_METER,
            headers=structured_headers,
        )

        assert dataset.metadata.name == "Seismic 3D"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "line"
        assert dataset.metadata.attributes["processingStage"] == "post-stack"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, "depth")

        # Verify seismic variable
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("inline", 256), ("crossline", 512), ("depth", 1024)],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.algorithm == "zstd"
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [128, 128, 128]
        assert seismic.metadata.stats_v1 is None

    def test_build_dataset_time(self, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DPostStackTimeTemplate build with time domain."""
        t = Seismic3DPostStackTemplate(domain="time")

        assert t.name == "PostStack3DTime"
        dataset = t.build_dataset(
            "Seismic 3D",
            sizes=[256, 512, 1024],
            horizontal_coord_unit=_UNIT_METER,
            headers=structured_headers,
        )

        assert dataset.metadata.name == "Seismic 3D"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "line"
        assert dataset.metadata.attributes["processingStage"] == "post-stack"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, "time")

        # Verify seismic variable
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("inline", 256), ("crossline", 512), ("time", 1024)],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.algorithm == "zstd"
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [128, 128, 128]
        assert seismic.metadata.stats_v1 is None
