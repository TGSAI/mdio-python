"""Unit tests for Seismic2DPostStackTemplate."""

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.v1 import dataset
from mdio.schemas.v1.dataset_builder import _get_named_dimension
from mdio.schemas.v1.stats import SummaryStatistics
from mdio.schemas.v1.templates.seismic_2d_poststack import Seismic2DPostStackTemplate

from mdio.schemas.v1.units import AllUnits, LengthUnitEnum, LengthUnitModel, TimeUnitEnum, TimeUnitModel
from tests.unit.v1.helpers import validate_variable


_UNIT_METER = AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))
_UNIT_SECOND = AllUnits(units_v1=TimeUnitModel(time=TimeUnitEnum.SECOND))


class TestSeismic2DPostStackTemplate:
    """Unit tests for Seismic2DPostStackTemplate."""

    def test_configuration_depth(self):
        """Test configuration of Seismic2DPostStackTemplate with depth domain."""
        t = Seismic2DPostStackTemplate("depth")

        # Template attributes
        assert t._trace_domain == "depth"
        assert t._coord_dim_names == ["cdp"]
        assert t._dim_names == ["cdp", "depth"]
        assert t._coord_names == ["cdp-x", "cdp-y"]
        assert t._var_name == "StackedAmplitude"
        assert t._var_chunk_shape == [1024, 1024]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._coord_units == []

        # Verify dataset attributes
        attrs = t._load_dataset_attributes()
        assert attrs.attributes == {
            "surveyDimensionality": "2D",
            "ensembleType": "line",
            "processingStage": "post-stack"
        }

    def test_configuration_time(self):
        """Test configuration of Seismic2DPostStackTemplate with time domain."""
        t = Seismic2DPostStackTemplate("time")

        # Template attributes
        assert t._trace_domain == "time"
        assert t._coord_dim_names == ["cdp"]
        assert t._dim_names == ["cdp", "time"]
        assert t._coord_names == ["cdp-x", "cdp-y"]
        assert t._var_name == "StackedAmplitude"
        assert t._var_chunk_shape == [1024, 1024]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._coord_units == []

    def test_domain_case_handling(self):
        """Test that domain parameter handles different cases correctly."""
        # Test uppercase
        t1 = Seismic2DPostStackTemplate("ELEVATION")
        assert t1._trace_domain == "elevation"
        assert t1.get_name() == "PostStack2DElevation"
        
        # Test mixed case
        t2 = Seismic2DPostStackTemplate("elevatioN")
        assert t2._trace_domain == "elevation"
        assert t2.get_name() == "PostStack2DElevation"

    
    def test_build_dataset_depth(self):
        """Test building a complete 2D depth dataset."""
        t = Seismic2DPostStackTemplate("depth")

        dataset = t.build_dataset(
            "Seismic 2D Depth Line 001",
            sizes=[2048, 4096],
            coord_units=[_UNIT_METER, _UNIT_METER]  # Both coordinates and depth in meters
        )

        # Verify dataset metadata
        assert dataset.metadata.name == "Seismic 2D Depth Line 001"
        assert dataset.metadata.attributes["surveyDimensionality"] == "2D"
        assert dataset.metadata.attributes["ensembleType"] == "line"
        assert dataset.metadata.attributes["processingStage"] == "post-stack"

        # 2 coordinate variables + 1 data variable = 5 variables
        assert len(dataset.variables) == 3

        # Verify coordinate variables
        cdp_x = validate_variable(
            dataset,
            name="cdp-x",
            dims=[("cdp", 2048)],
            coords=["cdp-x"],
            dtype=ScalarType.FLOAT64,
        )
        assert cdp_x.metadata.units_v1.length == LengthUnitEnum.METER

        cdp_y = validate_variable(
            dataset,
            name="cdp-y",
            dims=[("cdp", 2048)],
            coords=["cdp-y"],
            dtype=ScalarType.FLOAT64,
        )
        assert cdp_y.metadata.units_v1.length == LengthUnitEnum.METER

        # Verify seismic variable
        seismic = validate_variable(
            dataset,
            name="StackedAmplitude",
            dims=[("cdp", 2048), ("depth", 4096)],
            coords=["cdp-x", "cdp-y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [1024, 1024]
        assert seismic.metadata.stats_v1 is None

    def test_build_dataset_time(self):
        """Test building a complete 2D time dataset."""
        t = Seismic2DPostStackTemplate("time")

        dataset = t.build_dataset(
            "Seismic 2D Time Line 001",
            sizes=[2048, 4096],
            coord_units=[_UNIT_METER, _UNIT_METER]  # Coordinates in meters, time in seconds
        )

        # Verify dataset metadata
        assert dataset.metadata.name == "Seismic 2D Time Line 001"
        assert dataset.metadata.attributes["surveyDimensionality"] == "2D"
        assert dataset.metadata.attributes["ensembleType"] == "line"
        assert dataset.metadata.attributes["processingStage"] == "post-stack"

        # Verify variables count
        assert len(dataset.variables) == 3

        # Verify coordinate variables
        v = validate_variable(
            dataset,
            name="cdp-x",
            dims=[("cdp", 2048)],
            coords=["cdp-x"],
            dtype=ScalarType.FLOAT64,
        )
        assert v.metadata.units_v1.length == LengthUnitEnum.METER

        v = validate_variable(
            dataset,
            name="cdp-y",
            dims=[("cdp", 2048)],
            coords=["cdp-y"],
            dtype=ScalarType.FLOAT64,
        )
        assert v.metadata.units_v1.length == LengthUnitEnum.METER

        # Verify seismic variable
        v = validate_variable(
            dataset,
            name="StackedAmplitude",
            dims=[("cdp", 2048), ("time", 4096)],
            coords=["cdp-x", "cdp-y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(v.metadata.chunk_grid, RegularChunkGrid)
        assert v.metadata.chunk_grid.configuration.chunk_shape == [1024, 1024]
        assert v.metadata.stats_v1 is None

    def test_time_vs_depth_comparison(self):
        """Test differences between time and depth templates."""
        time_template = Seismic2DPostStackTemplate("time")
        depth_template = Seismic2DPostStackTemplate("depth")
        
        # Different trace domains
        assert time_template._trace_domain == "time"
        assert depth_template._trace_domain == "depth"
        
        # Different names
        assert time_template.get_name() == "PostStack2DTime"
        assert depth_template.get_name() == "PostStack2DDepth"
        
        # Same other attributes
        assert time_template._coord_dim_names == depth_template._coord_dim_names
        assert time_template._coord_names == depth_template._coord_names
        assert time_template._var_name == depth_template._var_name
        assert time_template._var_chunk_shape == depth_template._var_chunk_shape



