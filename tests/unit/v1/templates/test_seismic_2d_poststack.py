"""Unit tests for Seismic2DPostStackTemplate."""

from tests.unit.v1.helpers import validate_variable

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.templates.seismic_2d_poststack import Seismic2DPostStackTemplate
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.units import TimeUnitEnum
from mdio.schemas.v1.units import TimeUnitModel

_UNIT_METER = AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))
_UNIT_SECOND = AllUnits(units_v1=TimeUnitModel(time=TimeUnitEnum.SECOND))


def _validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType) -> None:
    """Validate the coordinate, headers, trace_mask variables in the dataset."""
    # Verify variables
    # 1 dim coords + 2 non-dim coords + 1 data + 1 trace mask + 1 headers = 6 variables
    assert len(dataset.variables) == 6

    # Verify trace headers
    validate_variable(
        dataset,
        name="headers",
        dims=[("cdp", 2048)],
        coords=["cdp_x", "cdp_y"],
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[("cdp", 2048)],
        coords=["cdp_x", "cdp_y"],
        dtype=ScalarType.BOOL,
    )

    # Verify dimension coordinate variables
    inline = validate_variable(
        dataset,
        name="cdp",
        dims=[("cdp", 2048)],
        coords=["cdp"],
        dtype=ScalarType.INT32,
    )
    assert inline.metadata is None

    # Verify non-dimension coordinate variables
    cdp_x = validate_variable(
        dataset,
        name="cdp_x",
        dims=[("cdp", 2048)],
        coords=["cdp_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_x.metadata.units_v1.length == LengthUnitEnum.METER

    cdp_y = validate_variable(
        dataset,
        name="cdp_y",
        dims=[("cdp", 2048)],
        coords=["cdp_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert cdp_y.metadata.units_v1.length == LengthUnitEnum.METER


class TestSeismic2DPostStackTemplate:
    """Unit tests for Seismic2DPostStackTemplate."""

    def test_configuration_depth(self) -> None:
        """Test configuration of Seismic2DPostStackTemplate with depth domain."""
        t = Seismic2DPostStackTemplate("depth")

        # Template attributes
        assert t._trace_domain == "depth"
        assert t._coord_dim_names == ["cdp"]
        assert t._dim_names == ["cdp", "depth"]
        assert t._coord_names == ["cdp_x", "cdp_y"]
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
            "processingStage": "post-stack",
        }

        assert t.trace_variable_name == "amplitude"

    def test_configuration_time(self) -> None:
        """Test configuration of Seismic2DPostStackTemplate with time domain."""
        t = Seismic2DPostStackTemplate("time")

        # Template attributes
        assert t._trace_domain == "time"
        assert t._coord_dim_names == ["cdp"]
        assert t._dim_names == ["cdp", "time"]
        assert t._coord_names == ["cdp_x", "cdp_y"]
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
            "processingStage": "post-stack",
        }
        assert t.trace_variable_name == "amplitude"

    def test_domain_case_handling(self) -> None:
        """Test that domain parameter handles different cases correctly."""
        # Test uppercase
        t1 = Seismic2DPostStackTemplate("ELEVATION")
        assert t1._trace_domain == "elevation"
        assert t1.name == "PostStack2DElevation"

        # Test mixed case
        t2 = Seismic2DPostStackTemplate("elevatioN")
        assert t2._trace_domain == "elevation"
        assert t2.name == "PostStack2DElevation"

    def test_build_dataset_depth(self, structured_headers: StructuredType) -> None:
        """Test building a complete 2D depth dataset."""
        t = Seismic2DPostStackTemplate("depth")

        dataset = t.build_dataset(
            "Seismic 2D Depth Line 001",
            sizes=[2048, 4096],
            coord_units=[_UNIT_METER, _UNIT_METER],  # Both coordinates and depth in meters
            headers=structured_headers,
        )

        # Verify dataset metadata
        assert dataset.metadata.name == "Seismic 2D Depth Line 001"
        assert dataset.metadata.attributes["surveyDimensionality"] == "2D"
        assert dataset.metadata.attributes["ensembleType"] == "line"
        assert dataset.metadata.attributes["processingStage"] == "post-stack"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers)

        # Verify seismic variable
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("cdp", 2048), ("depth", 4096)],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [1024, 1024]
        assert seismic.metadata.stats_v1 is None

    def test_build_dataset_time(self, structured_headers: StructuredType) -> None:
        """Test building a complete 2D time dataset."""
        t = Seismic2DPostStackTemplate("time")

        dataset = t.build_dataset(
            "Seismic 2D Time Line 001",
            sizes=[2048, 4096],
            coord_units=[_UNIT_METER, _UNIT_METER],  # Coordinates in meters, time in seconds
            headers=structured_headers,
        )

        # Verify dataset metadata
        assert dataset.metadata.name == "Seismic 2D Time Line 001"
        assert dataset.metadata.attributes["surveyDimensionality"] == "2D"
        assert dataset.metadata.attributes["ensembleType"] == "line"
        assert dataset.metadata.attributes["processingStage"] == "post-stack"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers)

        # Verify seismic variable
        v = validate_variable(
            dataset,
            name="amplitude",
            dims=[("cdp", 2048), ("time", 4096)],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(v.metadata.chunk_grid, RegularChunkGrid)
        assert v.metadata.chunk_grid.configuration.chunk_shape == [1024, 1024]
        assert v.metadata.stats_v1 is None

    def test_time_vs_depth_comparison(self) -> None:
        """Test differences between time and depth templates."""
        time_template = Seismic2DPostStackTemplate("time")
        depth_template = Seismic2DPostStackTemplate("depth")

        # Different trace domains
        assert time_template._trace_domain == "time"
        assert depth_template._trace_domain == "depth"

        # Different names
        assert time_template.name == "PostStack2DTime"
        assert depth_template.name == "PostStack2DDepth"

        # Same other attributes
        assert time_template._coord_dim_names == depth_template._coord_dim_names
        assert time_template._coord_names == depth_template._coord_names
        assert time_template._var_chunk_shape == depth_template._var_chunk_shape
