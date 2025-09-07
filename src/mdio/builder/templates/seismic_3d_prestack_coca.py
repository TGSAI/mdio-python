"""Seismic3DPreStackCocaTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.units import AngleUnitModel
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.builder.templates.abstract_dataset_template import SeismicDataDomain


class Seismic3DPreStackCocaTemplate(AbstractDatasetTemplate):
    """Seismic Shot pre-stack 3D time or depth Dataset template."""

    def __init__(self, data_domain: SeismicDataDomain):
        super().__init__(data_domain=data_domain)

        self._coord_dim_names = ("inline", "crossline", "offset", "azimuth")
        self._dim_names = (*self._coord_dim_names, self._data_domain)
        self._coord_names = ("cdp_x", "cdp_y")
        self._var_chunk_shape = (8, 8, 32, 1, 1024)

    @property
    def _name(self) -> str:
        return f"PreStackCocaGathers3D{self._data_domain.capitalize()}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "common_offset_common_azimuth"}

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        self._builder.add_coordinate(
            "inline",
            dimensions=("inline",),
            data_type=ScalarType.INT32,
        )
        self._builder.add_coordinate(
            "crossline",
            dimensions=("crossline",),
            data_type=ScalarType.INT32,
        )
        self._builder.add_coordinate(
            "offset",
            dimensions=("offset",),
            data_type=ScalarType.INT32,
            metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
        )
        angle_unit = AngleUnitModel(angle="deg")
        self._builder.add_coordinate(
            "azimuth",
            dimensions=("azimuth",),
            data_type=ScalarType.FLOAT32,
            metadata=CoordinateMetadata(units_v1=angle_unit),
        )
        self._builder.add_coordinate(
            self.trace_domain,
            dimensions=(self.trace_domain,),
            data_type=ScalarType.INT32,
        )

        # Add non-dimension coordinates
        self._builder.add_coordinate(
            "cdp_x",
            dimensions=("inline", "crossline"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
        )
        self._builder.add_coordinate(
            "cdp_y",
            dimensions=("inline", "crossline"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
        )
