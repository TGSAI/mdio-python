"""Unit tests for Seismic3DPreStackShotTemplate."""

from tests.unit.v1.helpers import validate_variable

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.templates.seismic_3d_prestack_shot import Seismic3DPreStackShotTemplate
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
    # 4 dim coords + 5 non-dim coords + 1 data + 1 trace mask + 1 headers = 12 variables
    assert len(dataset.variables) == 12

    # Verify trace headers
    validate_variable(
        dataset,
        name="headers",
        dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["gun", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"],
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["gun", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"],
        dtype=ScalarType.BOOL,
    )

    # Verify dimension coordinate variables
    inline = validate_variable(
        dataset,
        name="shot_point",
        dims=[("shot_point", 256)],
        coords=["shot_point"],
        dtype=ScalarType.INT32,
    )
    assert inline.metadata is None

    crossline = validate_variable(
        dataset,
        name="cable",
        dims=[("cable", 512)],
        coords=["cable"],
        dtype=ScalarType.INT32,
    )
    assert crossline.metadata is None

    crossline = validate_variable(
        dataset,
        name="channel",
        dims=[("channel", 24)],
        coords=["channel"],
        dtype=ScalarType.INT32,
    )
    assert crossline.metadata is None

    domain = validate_variable(
        dataset,
        name=domain,
        dims=[(domain, 2048)],
        coords=[domain],
        dtype=ScalarType.INT32,
    )
    assert domain.metadata is None

    # Verify non-dimension coordinate variables
    validate_variable(
        dataset,
        name="gun",
        dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["gun"],
        dtype=ScalarType.UINT8,
    )

    source_coord_x = validate_variable(
        dataset,
        name="source_coord_x",
        dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["source_coord_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert source_coord_x.metadata.units_v1.length == LengthUnitEnum.METER

    source_coord_y = validate_variable(
        dataset,
        name="source_coord_y",
        dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["source_coord_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert source_coord_y.metadata.units_v1.length == LengthUnitEnum.METER

    group_coord_x = validate_variable(
        dataset,
        name="group_coord_x",
        dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["group_coord_x"],
        dtype=ScalarType.FLOAT64,
    )
    assert group_coord_x.metadata.units_v1.length == LengthUnitEnum.METER

    group_coord_y = validate_variable(
        dataset,
        name="group_coord_y",
        dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
        coords=["group_coord_y"],
        dtype=ScalarType.FLOAT64,
    )
    assert group_coord_y.metadata.units_v1.length == LengthUnitEnum.METER


class TestSeismic3DPreStackShotTemplate:
    """Unit tests for Seismic3DPreStackShotTemplate."""

    def test_configuration_depth(self) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate in depth domain."""
        t = Seismic3DPreStackShotTemplate(domain="DEPTH")

        # Template attributes for prestack shot
        assert t._trace_domain == "depth"
        assert t._coord_dim_names == ["shot_point", "cable", "channel"]
        assert t._dim_names == ["shot_point", "cable", "channel", "depth"]
        assert t._coord_names == ["gun", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"]
        assert t._var_chunk_shape == [1, 1, 512, 4096]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._horizontal_coord_unit is None

        # Verify prestack shot attributes
        attrs = t._load_dataset_attributes()
        assert attrs.attributes == {
            "surveyDimensionality": "3D",
            "ensembleType": "shot_point",
            "processingStage": "pre-stack",
        }
        assert t.trace_variable_name == "amplitude"

    def test_configuration_time(self) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate in time domain."""
        t = Seismic3DPreStackShotTemplate(domain="TIME")

        # Template attributes for prestack shot
        assert t._trace_domain == "time"
        assert t._coord_dim_names == ["shot_point", "cable", "channel"]
        assert t._dim_names == ["shot_point", "cable", "channel", "time"]
        assert t._coord_names == ["gun", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"]
        assert t._var_chunk_shape == [1, 1, 512, 4096]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._horizontal_coord_unit is None

        # Verify prestack shot attributes
        attrs = t._load_dataset_attributes()
        assert attrs.attributes == {
            "surveyDimensionality": "3D",
            "ensembleType": "shot_point",
            "processingStage": "pre-stack",
        }

        assert t.name == "PreStackShotGathers3DTime"

    def test_domain_case_handling(self) -> None:
        """Test that domain parameter handles different cases correctly."""
        # Test uppercase
        t1 = Seismic3DPreStackShotTemplate("ELEVATION")
        assert t1._trace_domain == "elevation"
        assert t1.name == "PreStackShotGathers3DElevation"

        # Test mixed case
        t2 = Seismic3DPreStackShotTemplate("elevatioN")
        assert t2._trace_domain == "elevation"
        assert t2.name == "PreStackShotGathers3DElevation"

    def test_build_dataset_depth(self, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate build in depth domain."""
        t = Seismic3DPreStackShotTemplate(domain="depth")

        assert t.name == "PreStackShotGathers3DDepth"
        dataset = t.build_dataset(
            "Gulf of Mexico 3D Shot Depth",
            sizes=[256, 512, 24, 2048],
            horizontal_coord_unit=_UNIT_METER,
            headers=structured_headers,
        )

        assert dataset.metadata.name == "Gulf of Mexico 3D Shot Depth"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "shot_point"
        assert dataset.metadata.attributes["processingStage"] == "pre-stack"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, "depth")

        # Verify seismic variable (prestack shot depth data)
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("shot_point", 256), ("cable", 512), ("channel", 24), ("depth", 2048)],
            coords=["gun", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.algorithm == "zstd"
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [1, 1, 512, 4096]
        assert seismic.metadata.stats_v1 is None

    def test_build_dataset_time(self, structured_headers: StructuredType) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate build in time domain."""
        t = Seismic3DPreStackShotTemplate(domain="time")

        assert t.name == "PreStackShotGathers3DTime"
        dataset = t.build_dataset(
            "North Sea 3D Shot Time",
            sizes=[256, 512, 24, 2048],
            horizontal_coord_unit=_UNIT_METER,
            headers=structured_headers,
        )

        assert dataset.metadata.name == "North Sea 3D Shot Time"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "shot_point"
        assert dataset.metadata.attributes["processingStage"] == "pre-stack"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, "time")

        # Verify seismic variable (prestack shot time data)
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("shot_point", 256), ("cable", 512), ("channel", 24), ("time", 2048)],
            coords=["gun", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.algorithm == "zstd"
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [1, 1, 512, 4096]
        assert seismic.metadata.stats_v1 is None
