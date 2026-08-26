"""Unit tests for Seismic3DSingleNodeContinuousReceiverGathersTemplate."""

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
from mdio.builder.templates.seismic_3d_single_node_continuous_receiver_gathers import (
    Seismic3DSingleNodeContinuousReceiverGathersTemplate,
)

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)

DATASET_SIZE_MAP = {
    "component": 1,
    "epoch": 2048,
    "time": 15001,
}
DATASET_DTYPE_MAP = {
    "component": "uint8",
    "epoch": "int64",
    "time": "int32",
}
EXPECTED_COORDINATES = ["group_coord_x", "group_coord_y"]
EXPECTED_CHUNK_SHAPE = (1, 140, 15001)


def _validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType, domain: str) -> None:
    """Validate the coordinate, headers, trace_mask variables in the dataset."""
    assert len(dataset.variables) == 8

    validate_variable(
        dataset,
        name="headers",
        dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k != domain],
        coords=EXPECTED_COORDINATES,
        dtype=headers,
    )

    validate_variable(
        dataset,
        name="trace_mask",
        dims=[(k, v) for k, v in DATASET_SIZE_MAP.items() if k != domain],
        coords=EXPECTED_COORDINATES,
        dtype=ScalarType.BOOL,
    )

    for dim_name, dim_size in DATASET_SIZE_MAP.items():
        validate_variable(
            dataset,
            name=dim_name,
            dims=[(dim_name, dim_size)],
            coords=[dim_name],
            dtype=ScalarType(DATASET_DTYPE_MAP[dim_name]),
        )

    for coord_name in EXPECTED_COORDINATES:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=[("component", DATASET_SIZE_MAP["component"])],
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER

    epoch = next(v for v in dataset.variables if v.name == "epoch")
    assert epoch.metadata.units_v1.time == TimeUnitEnum.MICROSECOND


class TestSeismic3DSingleNodeContinuousReceiverGathersTemplate:
    """Unit tests for Seismic3DSingleNodeContinuousReceiverGathersTemplate."""

    def test_configuration(self) -> None:
        """Test template configuration and attributes."""
        t = Seismic3DSingleNodeContinuousReceiverGathersTemplate()

        assert t.name == "SingleNodeContRecvrGathers"
        assert t._dim_names == ("component", "epoch", "time")
        assert t.calculated_dimension_names == ()
        assert t.synthesize_missing_dims == ("component",)
        assert t._physical_coord_names == ("group_coord_x", "group_coord_y")
        assert t._logical_coord_names == ()
        assert t._var_chunk_shape == EXPECTED_CHUNK_SHAPE
        assert t.declare_dim_coordinate_types()["epoch"] == ScalarType.INT64

        assert t._builder is None
        assert t._dim_sizes == ()

        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyType": "3D", "gatherType": "continuous_receiver"}
        assert t.default_variable_name == "amplitude"

    def test_chunk_shape(self) -> None:
        """Default chunk shape is (1, 140, 15001) over (component, epoch, time)."""
        t = Seismic3DSingleNodeContinuousReceiverGathersTemplate()

        assert t.full_chunk_shape == EXPECTED_CHUNK_SHAPE
        assert 1 * 140 * 15001 * 4 == 8_400_560  # ~8 MiB

    def test_build_dataset(self, structured_headers: StructuredType) -> None:
        """Test building a complete dataset with the template."""
        t = Seismic3DSingleNodeContinuousReceiverGathersTemplate()

        t.add_units({"group_coord_x": UNITS_METER, "group_coord_y": UNITS_METER})
        t.add_units({"time": UNITS_SECOND})

        sizes = tuple(DATASET_SIZE_MAP.values())
        dataset = t.build_dataset("SingleNodeSurvey", sizes=sizes, header_dtype=structured_headers)

        assert dataset.metadata.name == "SingleNodeSurvey"
        assert dataset.metadata.attributes["surveyType"] == "3D"
        assert dataset.metadata.attributes["gatherType"] == "continuous_receiver"
        assert dataset.metadata.attributes["defaultVariableName"] == "amplitude"

        _validate_coordinates_headers_trace_mask(dataset, structured_headers, "time")

        seismic = validate_variable(
            dataset,
            name="amplitude",
            dims=list(DATASET_SIZE_MAP.items()),
            coords=EXPECTED_COORDINATES,
            dtype=ScalarType.FLOAT32,
        )
        assert isinstance(seismic.compressor, Blosc)
        assert seismic.compressor.cname == BloscCname.zstd
        assert isinstance(seismic.metadata.chunk_grid, RegularChunkGrid)
        assert seismic.metadata.chunk_grid.configuration.chunk_shape == EXPECTED_CHUNK_SHAPE
        assert seismic.metadata.stats_v1 is None

    def test_depth_domain_rejected(self) -> None:
        """Depth domain is rejected."""
        with pytest.raises(ValueError, match="only supports the time domain"):
            Seismic3DSingleNodeContinuousReceiverGathersTemplate(data_domain="depth")
