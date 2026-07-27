"""Seismic3DCrgReceiverGathersTemplate MDIO v1 dataset template.

OBN Continuous Receiver Gathers (CRG), clock-time product. Each ocean-bottom node
records continuously; the delivery bundles many receivers, and every receiver's
long recording is segmented into fixed-length (e.g. 30 s) traces.
"""

from typing import Any

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import CoordinateSpec
from mdio.builder.templates.types import DimCoordinateTypes
from mdio.builder.templates.types import SeismicDataDomain


class Seismic3DCrgReceiverGathersTemplate(AbstractDatasetTemplate):
    """Seismic 3D OBN Continuous Receiver Gathers (clock time) template.

    The template declares the spatial receiver geometry ``(component, receiver_line,
    receiver)`` plus the vertical axis. The per-receiver segment axis is **not** declared
    here: it is inserted at ingest as a ``trace`` dimension by the ``HasDuplicates`` grid
    override, which appends a per-``(component, receiver_line, receiver)`` counter in trace
    order. This reuses the shipping duplicate-handling machinery instead of adding a new
    calculated dimension, so it stays backwards compatible with the existing ``trace``
    handling downstream. Size the inserted ``trace`` chunk (and, for very long campaigns,
    its dtype) through ``GridOverrides(has_duplicates=True, chunksize=..., trace_dtype=...)``.

    Real recording time is preserved per trace in ``headers`` (an ``epoch`` field, when the
    SEG-Y spec includes it), not baked into the segment index; the segment axis is a dense
    positional index in acquisition order.

    Special handling for the component dimension:
        If the SEG-Y spec does not contain a ``component`` field, the ingestion process
        synthesizes a ``component`` dimension with constant value 1 for all traces (a
        warning is logged). This is driven by ``synthesize_missing_dims`` and handled by
        ``ComponentSynthesisStrategy``, so one template ingests both single- and
        multi-component CRG data.
    """

    def __init__(self, data_domain: SeismicDataDomain = "time"):
        super().__init__(data_domain=data_domain)

        # The segment axis (`trace`) is inserted at ingest by HasDuplicates, so it is
        # deliberately absent from the declared dimensions here.
        self._spatial_dim_names = ("component", "receiver_line", "receiver")
        self.synthesize_missing_dims = ("component",)
        self._dim_names = (*self._spatial_dim_names, self._data_domain)
        self._physical_coord_names = ("group_coord_x", "group_coord_y")
        self._logical_coord_names = ()
        # 3 spatial + vertical. `time` holds the whole trace in one chunk (>= 15001
        # samples); the HasDuplicates effect inserts the segment chunk between them.
        self._var_chunk_shape = (1, 1, 1, 16384)

    @property
    def _name(self) -> str:
        return "ObnContinuousReceiverGathers3D"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "common_receiver"}

    def declare_dim_coordinate_types(self) -> DimCoordinateTypes:
        """Declare the data types for each dimension coordinate in this template."""
        return {
            "component": ScalarType.UINT8,
            "receiver_line": ScalarType.UINT32,
            "receiver": ScalarType.UINT32,
            self._data_domain: ScalarType.INT32,
        }

    def declare_coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        """Declare the receiver-indexed physical coordinates for this template."""
        receiver_dims = ("receiver_line", "receiver")
        return (
            CoordinateSpec(name="group_coord_x", dimensions=receiver_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="group_coord_y", dimensions=receiver_dims, dtype=ScalarType.FLOAT64),
        )

    def _add_coordinates(self) -> None:
        # Add dimension coordinates.
        for name in ("component", "receiver_line", "receiver"):
            self._builder.add_coordinate(
                name,
                dimensions=(name,),
                data_type=self._dim_dtype(name),
            )
        self._builder.add_coordinate(
            self._data_domain,
            dimensions=(self._data_domain,),
            data_type=self._dim_dtype(self._data_domain),
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(self._data_domain)),
        )

        # Add non-dimension coordinates: receiver X/Y indexed by (receiver_line, receiver).
        for name in ("group_coord_x", "group_coord_y"):
            self._builder.add_coordinate(
                name,
                dimensions=("receiver_line", "receiver"),
                data_type=ScalarType.FLOAT64,
                metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(name)),
            )
