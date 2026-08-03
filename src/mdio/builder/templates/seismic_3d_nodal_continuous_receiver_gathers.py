"""Seismic3DNodalContinuousReceiverGathersTemplate MDIO v1 dataset template."""

from typing import Any

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.units import TimeUnitEnum
from mdio.builder.schemas.v1.units import TimeUnitModel
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import CoordinateSpec
from mdio.builder.templates.types import DimCoordinateTypes
from mdio.builder.templates.types import SeismicDataDomain

EPOCH_UNIT = TimeUnitModel(time=TimeUnitEnum.MICROSECOND)


class Seismic3DNodalContinuousReceiverGathersTemplate(AbstractDatasetTemplate):
    """Seismic 3D continuous receiver gathers template (land nodes and ocean-bottom nodes).

    Continuously recording receivers have no shots to grid on. Each SEG-Y trace is a
    fixed-length segment of one receiver's recording. ``segment_index`` is a calculated
    dimension (like ``shot_index`` on ``ObnReceiverGathers3D``), filled at ingest by
    ``GridOverrides(calculate_segment_index=True)``. The override ranks each trace's
    ``epoch`` within its ``(receiver_line, receiver, component)`` group.

    Original ``epoch`` values are preserved as an ``int64`` microsecond coordinate spanning
    the full spatial key. Empty cells (receivers with fewer segments than the grid axis)
    hold the int64 fill sentinel and must be excluded via ``trace_mask``.

    Special handling for the component dimension:
        If the SEG-Y spec does not contain a ``component`` field, ingestion synthesizes the
        dimension with constant value 1 for all traces and logs a warning. This is driven by
        ``synthesize_missing_dims`` and handled by ``ComponentSynthesisStrategy``.
    """

    synthesize_missing_dims = ("component",)

    def __init__(self, data_domain: SeismicDataDomain = "time"):
        if data_domain != "time":
            msg = "NodalContinuousReceiverGathers3D only supports the time domain, got {data_domain!r}"
            raise ValueError(msg.format(data_domain=data_domain))

        super().__init__(data_domain=data_domain)

        self._spatial_dim_names = ("receiver_line", "receiver", "component", "segment_index")
        self._calculated_dims = ("segment_index",)
        self._dim_names = (*self._spatial_dim_names, self._data_domain)
        self._physical_coord_names = ("group_coord_x", "group_coord_y")
        self._logical_coord_names = ("epoch",)
        self._var_chunk_shape = (1, 1, 1, 180, 15001)
        self.add_units({"epoch": EPOCH_UNIT})

    @property
    def _name(self) -> str:
        return "NodalContinuousReceiverGathers3D"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "continuous_receiver"}

    def declare_coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        """Declare receiver- and segment-indexed coordinates for continuous receiver gathers."""
        receiver_dims = ("receiver_line", "receiver")
        return (
            CoordinateSpec(name="group_coord_x", dimensions=receiver_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="group_coord_y", dimensions=receiver_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="epoch", dimensions=self.spatial_dimension_names, dtype=ScalarType.INT64),
        )

    def declare_dim_coordinate_types(self) -> DimCoordinateTypes:
        """Declare the data types for each dimension coordinate in this template."""
        return {
            "receiver_line": ScalarType.UINT32,
            "receiver": ScalarType.UINT32,
            "component": ScalarType.UINT8,
            self._data_domain: ScalarType.INT32,
        }

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        # EXCLUDE: `segment_index` since it's 0-N (calculated dimension)
        for name in ("receiver_line", "receiver", "component", self._data_domain):
            self._add_dimension_coordinate(name)

        # Add non-dimension coordinates
        for name in ("group_coord_x", "group_coord_y"):
            self._builder.add_coordinate(
                name,
                dimensions=("receiver_line", "receiver"),
                data_type=ScalarType.FLOAT64,
                metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(name)),
            )

        self._builder.add_coordinate(
            "epoch",
            dimensions=self.spatial_dimension_names,
            data_type=ScalarType.INT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("epoch")),
        )
