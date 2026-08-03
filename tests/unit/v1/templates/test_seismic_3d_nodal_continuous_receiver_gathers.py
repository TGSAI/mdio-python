"""Unit tests for Seismic3DNodalContinuousReceiverGathersTemplate."""

import numpy as np
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
from mdio.builder.templates.seismic_3d_nodal_continuous_receiver_gathers import (
    Seismic3DNodalContinuousReceiverGathersTemplate,
)
from mdio.core.utils_write import MAX_COORDINATES_BYTES
from mdio.core.utils_write import get_constrained_chunksize
from mdio.ingestion.dataset_factory import build_mdio_dataset
from mdio.ingestion.schema.resolver import SchemaResolver

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)

# Typical continuous receiver deliverable scale used by chunking tests.
DATASET_SIZE_MAP = {
    "receiver_line": 1,
    "receiver": 232,
    "component": 1,
    "segment_index": 27108,
    "time": 15001,
}
DATASET_DTYPE_MAP = {
    "receiver_line": "uint32",
    "receiver": "uint32",
    "component": "uint8",
    "time": "int32",
}
EXPECTED_COORDINATES = ["group_coord_x", "group_coord_y", "epoch"]
RECEIVER_DIMS = ("receiver_line", "receiver")
EPOCH_DIMS = ("receiver_line", "receiver", "component", "segment_index")
EXPECTED_CHUNK_SHAPE = (1, 1, 1, 180, 15001)


def _validate_coordinates_headers_trace_mask(dataset: Dataset, headers: StructuredType, domain: str) -> None:
    """Validate the coordinate, headers, trace_mask variables in the dataset."""
    # 4 dim coords (excl. segment_index which is 0-N) + 3 non-dim coords + 1 data + 1 trace mask
    # + 1 headers = 10 variables
    assert len(dataset.variables) == 10

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

    # Verify dimension coordinate variables (excluding segment_index which is calculated 0-N)
    for dim_name, dim_size in DATASET_SIZE_MAP.items():
        if dim_name == "segment_index":
            continue
        validate_variable(
            dataset,
            name=dim_name,
            dims=[(dim_name, dim_size)],
            coords=[dim_name],
            dtype=ScalarType(DATASET_DTYPE_MAP[dim_name]),
        )

    # Verify receiver coordinate variables (indexed by receiver only)
    for coord_name in ["group_coord_x", "group_coord_y"]:
        coord = validate_variable(
            dataset,
            name=coord_name,
            dims=[(k, DATASET_SIZE_MAP[k]) for k in RECEIVER_DIMS],
            coords=[coord_name],
            dtype=ScalarType.FLOAT64,
        )
        assert coord.metadata.units_v1.length == LengthUnitEnum.METER

    # Verify epoch coordinate (indexed by full spatial key)
    epoch = validate_variable(
        dataset,
        name="epoch",
        dims=[(k, DATASET_SIZE_MAP[k]) for k in EPOCH_DIMS],
        coords=["epoch"],
        dtype=ScalarType.INT64,
    )
    assert epoch.metadata.units_v1.time == TimeUnitEnum.MICROSECOND


class TestSeismic3DNodalContinuousReceiverGathersTemplate:
    """Unit tests for Seismic3DNodalContinuousReceiverGathersTemplate."""

    def test_configuration(self) -> None:
        """Test template configuration and attributes."""
        t = Seismic3DNodalContinuousReceiverGathersTemplate()

        assert t.name == "NodalContinuousReceiverGathers3D"
        assert t._dim_names == ("receiver_line", "receiver", "component", "segment_index", "time")
        assert t._calculated_dims == ("segment_index",)
        assert t._physical_coord_names == ("group_coord_x", "group_coord_y")
        assert t._logical_coord_names == ("epoch",)
        assert t._var_chunk_shape == (1, 1, 1, 180, 15001)

        # Variables instantiated when build_dataset() is called
        assert t._builder is None
        assert t._dim_sizes == ()

        attrs = t._load_dataset_attributes()
        assert attrs == {"surveyType": "3D", "gatherType": "continuous_receiver"}
        assert t.default_variable_name == "amplitude"

    def test_component_is_synthesized_when_missing(self) -> None:
        """Component is listed in synthesize_missing_dims."""
        t = Seismic3DNodalContinuousReceiverGathersTemplate()

        assert t.synthesize_missing_dims == ("component",)

    def test_segment_index_is_calculated(self) -> None:
        """segment_index is a calculated dimension with no dim coordinate type."""
        t = Seismic3DNodalContinuousReceiverGathersTemplate()

        assert t.calculated_dimension_names == ("segment_index",)
        assert "segment_index" not in t.declare_dim_coordinate_types()

    def test_chunk_matches_ninety_minute_segment_budget(self) -> None:
        """Default chunk shape matches EXPECTED_CHUNK_SHAPE and exceeds 8 MiB float32."""
        t = Seismic3DNodalContinuousReceiverGathersTemplate()
        t._dim_sizes = tuple(DATASET_SIZE_MAP.values())

        chunk_shape = t.full_chunk_shape
        assert chunk_shape == EXPECTED_CHUNK_SHAPE

        chunk_bytes = int(np.prod(chunk_shape)) * 4
        assert chunk_bytes == 180 * 15001 * 4
        assert chunk_bytes > 8 * 1024 * 1024

    def test_chunking_keeps_one_receiver_packs_segments(self) -> None:
        """Chunk keeps one receiver/component; packs 180 segments and 15001 samples."""
        t = Seismic3DNodalContinuousReceiverGathersTemplate()

        chunk_shape = t._var_chunk_shape
        assert chunk_shape[0] == 1  # receiver_line
        assert chunk_shape[1] == 1  # receiver
        assert chunk_shape[2] == 1  # component
        assert chunk_shape[3] == 180  # segment_index
        assert chunk_shape[4] == 15001  # time

    def test_build_dataset(self, structured_headers: StructuredType) -> None:
        """Test building a complete dataset with the template."""
        t = Seismic3DNodalContinuousReceiverGathersTemplate()

        t.add_units({"group_coord_x": UNITS_METER, "group_coord_y": UNITS_METER})
        t.add_units({"time": UNITS_SECOND})

        sizes = tuple(DATASET_SIZE_MAP.values())
        dataset = t.build_dataset("ContinuousSurvey3D", sizes=sizes, header_dtype=structured_headers)

        assert dataset.metadata.name == "ContinuousSurvey3D"
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
            Seismic3DNodalContinuousReceiverGathersTemplate(data_domain="depth")

    def test_epoch_chunked_under_coordinate_cap_at_advertised_scale(self) -> None:
        """Epoch chunking stays under MAX_COORDINATES_BYTES at production scale."""
        template = Seismic3DNodalContinuousReceiverGathersTemplate()
        schema = SchemaResolver().resolve(template)
        sizes = tuple(DATASET_SIZE_MAP.values())
        dataset = build_mdio_dataset(schema=schema, sizes=sizes)

        epoch = next(v for v in dataset.variables if v.name == "epoch")
        assert epoch.metadata is not None
        assert epoch.metadata.chunk_grid is not None
        chunk_shape = epoch.metadata.chunk_grid.configuration.chunk_shape

        expected = get_constrained_chunksize(
            shape=tuple(DATASET_SIZE_MAP[k] for k in EPOCH_DIMS),
            dtype="int64",
            max_bytes=MAX_COORDINATES_BYTES,
        )
        assert chunk_shape == expected
        assert int(np.prod(chunk_shape)) * 8 <= MAX_COORDINATES_BYTES
