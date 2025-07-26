"""Unit tests for Seismic3DPreStackShotTemplate."""

from tests.unit.v1.helpers import validate_variable

from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.v1.templates.seismic_3d_prestack_shot import Seismic3DPreStackShotTemplate
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.units import TimeUnitEnum
from mdio.schemas.v1.units import TimeUnitModel

_UNIT_METER = AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))
_UNIT_SECOND = AllUnits(units_v1=TimeUnitModel(time=TimeUnitEnum.SECOND))


class TestSeismic3DPreStackShotTemplate:
    """Unit tests for Seismic3DPreStackShotTemplate."""

    def test_configuration_depth(self) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate in depth domain."""
        t = Seismic3DPreStackShotTemplate(domain="DEPTH")

        # Template attributes for prestack shot
        assert t._trace_domain == "depth"
        assert t._coord_dim_names == []
        assert t._dim_names == ["shot_point", "cable", "channel", "depth"]
        assert t._coord_names == ["gun", "shot-x", "shot-y", "receiver-x", "receiver-y"]
        assert t._var_chunk_shape == [1, 1, 512, 4096]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._coord_units == []

        # Verify prestack shot attributes
        attrs = t._load_dataset_attributes()
        assert attrs.attributes == {
            "surveyDimensionality": "3D",
            "ensembleType": "shot",
            "processingStage": "pre-stack",
        }
        assert t.get_data_variable_name() == "amplitude"

    def test_configuration_time(self) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate in time domain."""
        t = Seismic3DPreStackShotTemplate(domain="TIME")

        # Template attributes for prestack shot
        assert t._trace_domain == "time"
        assert t._coord_dim_names == []
        assert t._dim_names == ["shot_point", "cable", "channel", "time"]
        assert t._coord_names == ["gun", "shot-x", "shot-y", "receiver-x", "receiver-y"]
        assert t._var_chunk_shape == [1, 1, 512, 4096]

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == []
        assert t._coord_units == []

        # Verify prestack shot attributes
        attrs = t._load_dataset_attributes()
        assert attrs.attributes == {
            "surveyDimensionality": "3D",
            "ensembleType": "shot",
            "processingStage": "pre-stack",
        }

        assert t.get_name() == "PreStackShotGathers3DTime"

    def test_domain_case_handling(self) -> None:
        """Test that domain parameter handles different cases correctly."""
        # Test uppercase
        t1 = Seismic3DPreStackShotTemplate("ELEVATION")
        assert t1._trace_domain == "elevation"
        assert t1.get_name() == "PreStackShotGathers3DElevation"

        # Test mixed case
        t2 = Seismic3DPreStackShotTemplate("elevatioN")
        assert t2._trace_domain == "elevation"
        assert t2.get_name() == "PreStackShotGathers3DElevation"

    def test_build_dataset_depth(self) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate build in depth domain."""
        t = Seismic3DPreStackShotTemplate(domain="depth")

        assert t.get_name() == "PreStackShotGathers3DDepth"
        dataset = t.build_dataset(
            "Gulf of Mexico 3D Shot Depth",
            sizes=[256, 512, 24, 2048],
            coord_units=[_UNIT_METER, _UNIT_METER],
        )

        assert dataset.metadata.name == "Gulf of Mexico 3D Shot Depth"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "shot"
        assert dataset.metadata.attributes["processingStage"] == "pre-stack"

        # Verify variables (including dimension variables)
        # 5 coordinate variables + 1 data variable + 1 trace mask = 7 variables
        assert len(dataset.variables) == 7

        # Verify coordinate variables
        validate_variable(
            dataset,
            name="gun",
            dims=[("shot_point", 256)],
            coords=["gun"],
            dtype=ScalarType.UINT8,
        )

        shot_x = validate_variable(
            dataset,
            name="shot-x",
            dims=[("shot_point", 256)],
            coords=["shot-x"],
            dtype=ScalarType.FLOAT64,
        )
        assert shot_x.metadata.units_v1.length == LengthUnitEnum.METER

        shot_y = validate_variable(
            dataset,
            name="shot-y",
            dims=[("shot_point", 256)],
            coords=["shot-y"],
            dtype=ScalarType.FLOAT64,
        )
        assert shot_y.metadata.units_v1.length == LengthUnitEnum.METER

        receiver_x = validate_variable(
            dataset,
            name="receiver-x",
            dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
            coords=["receiver-x"],
            dtype=ScalarType.FLOAT64,
        )
        assert receiver_x.metadata.units_v1.length == LengthUnitEnum.METER

        receiver_y = validate_variable(
            dataset,
            name="receiver-y",
            dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
            coords=["receiver-y"],
            dtype=ScalarType.FLOAT64,
        )
        assert receiver_y.metadata.units_v1.length == LengthUnitEnum.METER

        # Verify seismic variable (prestack shot depth data)
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("shot_point", 256), ("cable", 512), ("channel", 24), ("depth", 2048)],
            coords=["gun", "shot-x", "shot-y", "receiver-x", "receiver-y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.algorithm == "zstd"
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [1, 1, 512, 4096]
        assert seismic.metadata.stats_v1 is None

        # TODO: Validate trace mask

    def test_build_dataset_time(self) -> None:
        """Unit tests for Seismic3DPreStackShotTemplate build in time domain."""
        t = Seismic3DPreStackShotTemplate(domain="time")

        assert t.get_name() == "PreStackShotGathers3DTime"
        dataset = t.build_dataset(
            "North Sea 3D Shot Time",
            sizes=[256, 512, 24, 2048],
            coord_units=[_UNIT_METER, _UNIT_METER],
        )

        assert dataset.metadata.name == "North Sea 3D Shot Time"
        assert dataset.metadata.attributes["surveyDimensionality"] == "3D"
        assert dataset.metadata.attributes["ensembleType"] == "shot"
        assert dataset.metadata.attributes["processingStage"] == "pre-stack"

        # Verify variables (including dimension variables)
        # 5 coordinate variables + 1 data variable + 1 trace mask = 7 variables
        assert len(dataset.variables) == 7

        # Verify coordinate variables
        validate_variable(
            dataset,
            name="gun",
            dims=[("shot_point", 256)],
            coords=["gun"],
            dtype=ScalarType.UINT8,
        )

        shot_x = validate_variable(
            dataset,
            name="shot-x",
            dims=[("shot_point", 256)],
            coords=["shot-x"],
            dtype=ScalarType.FLOAT64,
        )
        assert shot_x.metadata.units_v1.length == LengthUnitEnum.METER

        shot_y = validate_variable(
            dataset,
            name="shot-y",
            dims=[("shot_point", 256)],
            coords=["shot-y"],
            dtype=ScalarType.FLOAT64,
        )
        assert shot_y.metadata.units_v1.length == LengthUnitEnum.METER

        receiver_x = validate_variable(
            dataset,
            name="receiver-x",
            dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
            coords=["receiver-x"],
            dtype=ScalarType.FLOAT64,
        )
        assert receiver_x.metadata.units_v1.length == LengthUnitEnum.METER

        receiver_y = validate_variable(
            dataset,
            name="receiver-y",
            dims=[("shot_point", 256), ("cable", 512), ("channel", 24)],
            coords=["receiver-y"],
            dtype=ScalarType.FLOAT64,
        )
        assert receiver_y.metadata.units_v1.length == LengthUnitEnum.METER

        # Verify seismic variable (prestack shot time data)
        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=[("shot_point", 256), ("cable", 512), ("channel", 24), ("time", 2048)],
            coords=["gun", "shot-x", "shot-y", "receiver-x", "receiver-y"],
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.algorithm == "zstd"
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == [1, 1, 512, 4096]
        assert seismic.metadata.stats_v1 is None

        # TODO: Validate trace mask