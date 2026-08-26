"""Seismic3DSingleNodeContinuousReceiverGathersTemplate MDIO v1 dataset template."""

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


class Seismic3DSingleNodeContinuousReceiverGathersTemplate(AbstractDatasetTemplate):
    """Single-node continuous receiver gathers template.

    Dimensions are ``component``, ``epoch`` (int64 microseconds), and ``time``.
    Physical coordinates ``group_coord_x`` / ``group_coord_y`` are indexed by ``component``.

    Special handling for the component dimension:
        If the SEG-Y spec does not contain a ``component`` field, ingestion synthesizes the
        dimension with constant value 1 for all traces and logs a warning. This is driven by
        ``synthesize_missing_dims`` and handled by ``ComponentSynthesisStrategy``.
    """

    def __init__(self, data_domain: SeismicDataDomain = "time"):
        if data_domain != "time":
            msg = "SingleNodeContRecvrGathers only supports the time domain, got {data_domain!r}"
            raise ValueError(msg.format(data_domain=data_domain))

        super().__init__(data_domain=data_domain)

        self._dim_names = ("component", "epoch", self._data_domain)
        self.synthesize_missing_dims = ("component",)
        self._physical_coord_names = ("group_coord_x", "group_coord_y")
        self._logical_coord_names = ()
        self._var_chunk_shape = (1, 140, 15001)
        self.add_units({"epoch": EPOCH_UNIT})

    @property
    def _name(self) -> str:
        return "SingleNodeContRecvrGathers"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "continuous_receiver"}

    def declare_coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        """Declare node-position coordinates for single-node continuous gathers."""
        return (
            CoordinateSpec(name="group_coord_x", dimensions=("component",), dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="group_coord_y", dimensions=("component",), dtype=ScalarType.FLOAT64),
        )

    def declare_dim_coordinate_types(self) -> DimCoordinateTypes:
        """Declare the data types for each dimension coordinate in this template."""
        return {
            "component": ScalarType.UINT8,
            "epoch": ScalarType.INT64,
            self._data_domain: ScalarType.INT32,
        }

    def _add_coordinates(self) -> None:
        for name in ("component", "epoch", self._data_domain):
            self._add_dimension_coordinate(name)

        for name in ("group_coord_x", "group_coord_y"):
            self._builder.add_coordinate(
                name,
                dimensions=("component",),
                data_type=ScalarType.FLOAT64,
                metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(name)),
            )
