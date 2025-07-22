"""Unit tests for Seismic3DPostStackTemplate."""

from tests.unit.v1.helpers import validate_variable

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.v1.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.units import TimeUnitEnum
from mdio.schemas.v1.units import TimeUnitModel

_UNIT_METER = AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))
_UNIT_SECOND = AllUnits(units_v1=TimeUnitModel(time=TimeUnitEnum.SECOND))


class TestSeismic3DPostStackTemplate:
    """Unit tests for Seismic3DPostStackTemplate."""

    def test_configuration_depth(self) -> None:
        """Unit tests for Seismic3DPostStackTemplate with depth domain."""
        t = Seismic3DPostStackTemplate(domain="depth")

        # Template attributes to be overridden by subclasses
        assert t._trace_domain == "depth"  # Domain should be lowercased
        assert t._coord_dim_names == ["inline", "crossline"]
        assert t._dim_names == ["inline", "crossline", "depth"]
        assert t._coord_names == ["cdp-x", "cdp-y"]
        assert t._var_chunk_shape == [128, 128, 128]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._coord_units == []

        assert t._load_dataset_attributes().attributes == {
            "surveyDimensionality": "3D",
            "ensembleType": "line",
            "processingStage": "post-stack",
        }

        assert t.get_name() == "PostStack3DDepth"

    def test_configuration_time(self) -> None:
        """Unit tests for Seismic3DPostStackTemplate with time domain."""
        t = Seismic3DPostStackTemplate(domain="time")

        # Template attributes to be overridden by subclasses
        assert t._trace_domain == "time"  # Domain should be lowercased
        assert t._coord_dim_names == ["inline", "crossline"]
        assert t._dim_names == ["inline", "crossline", "time"]
        assert t._coord_names == ["cdp-x", "cdp-y"]
        assert t._var_chunk_shape == [128, 128, 128]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._coord_units == []

        assert t._load_dataset_attributes().attributes == {
            "surveyDimensionality": "3D",
            "ensembleType": "line",
            "processingStage": "post-stack",
        }

        assert t.get_name() == "PostStack3DTime"

    def test_domain_case_handling(self) -> None:
        """Test that domain parameter handles different cases correctly."""
        # Test uppercase
        t1 = Seismic3DPostStackTemplate("ELEVATION")
        assert t1._trace_domain == "elevation"
        assert t1.get_name() == "PostStack3DElevation"

        # Test mixed case
        t2 = Seismic3DPostStackTemplate("elevatioN")
        assert t2._trace_domain == "elevation"
        assert t2.get_name() == "PostStack3DElevation"

    def test_build_dataset_depth(self) -> None:
        """Unit tests for Seismic3DPostStackTemplate build with depth domain."""
        t = Seismic3DPostStackTemplate(domain="depth")

        assert t.get_name() == "PostStack3DDepth"
        dataset = t.build_dataset(
            "Seismic 3D", sizes=[256, 512, 1024], coord_units=[_UNIT_METER, _UNIT_METER]
        )

        assert dataset.metadata.name == "Seismic 3D"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "line"
        assert dataset.metadata.attributes["processingStage"] == "post-stack"

        # Verify variables
        # 2 coordinate variables + 1 data variables = 3 variables
        assert len(dataset.variables) == 3

        # Verify coordinate variables
        cdp_x = validate_variable(
            dataset,
            name="cdp-x",
            dims=[("inline", 256), ("crossline", 512)],
            coords=["cdp-x"],
            dtype=ScalarType.FLOAT64,
        )
        assert cdp_x.metadata.units_v1.length == LengthUnitEnum.METER

        cdp_y = validate_variable(
            dataset,
            name="cdp-y",
            dims=[("inline", 256), ("crossline", 512)],
            coords=["cdp-y"],
            dtype=ScalarType.FLOAT64,
        )
        assert cdp_y.metadata.units_v1.length == LengthUnitEnum.METER

        # Verify seismic variable
        seismic = validate_variable(
            dataset,
            name="StackedAmplitude",
            dims=[("inline", 256), ("crossline", 512), ("depth", 1024)],
            coords=["cdp-x", "cdp-y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.algorithm == "zstd"
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [128, 128, 128]
        assert seismic.metadata.stats_v1 is None

    def test_build_dataset_time(self) -> None:
        """Unit tests for Seismic3DPostStackTimeTemplate build with time domain."""
        t = Seismic3DPostStackTemplate(domain="time")

        assert t.get_name() == "PostStack3DTime"
        dataset = t.build_dataset(
            "Seismic 3D", sizes=[256, 512, 1024], coord_units=[_UNIT_METER, _UNIT_METER]
        )

        assert dataset.metadata.name == "Seismic 3D"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "line"
        assert dataset.metadata.attributes["processingStage"] == "post-stack"

        # Verify variables
        # 2 coordinate variables + 1 data variables = 3 variables
        assert len(dataset.variables) == 3

        # Verify coordinate variables
        cdp_x = validate_variable(
            dataset,
            name="cdp-x",
            dims=[("inline", 256), ("crossline", 512)],
            coords=["cdp-x"],
            dtype=ScalarType.FLOAT64,
        )
        assert cdp_x.metadata.units_v1.length == LengthUnitEnum.METER

        cdp_y = validate_variable(
            dataset,
            name="cdp-y",
            dims=[("inline", 256), ("crossline", 512)],
            coords=["cdp-y"],
            dtype=ScalarType.FLOAT64,
        )
        assert cdp_y.metadata.units_v1.length == LengthUnitEnum.METER

        # Verify seismic variable
        seismic = validate_variable(
            dataset,
            name="StackedAmplitude",
            dims=[("inline", 256), ("crossline", 512), ("time", 1024)],
            coords=["cdp-x", "cdp-y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.algorithm == "zstd"
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [128, 128, 128]
        assert seismic.metadata.stats_v1 is None
