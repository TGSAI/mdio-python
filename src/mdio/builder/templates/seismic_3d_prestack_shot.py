"""Seismic3DPreStackShotTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.builder.templates.abstract_dataset_template import SeismicDataDomain


class Seismic3DPreStackShotTemplate(AbstractDatasetTemplate):
    """Seismic Shot pre-stack 3D time or depth Dataset template."""

    def __init__(self, data_domain: SeismicDataDomain):
        super().__init__(data_domain=data_domain)

        self._coord_dim_names = ("shot_point", "cable", "channel")
        self._dim_names = (*self._coord_dim_names, self._data_domain)
        self._coord_names = ("gun", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y")
        self._var_chunk_shape = (8, 2, 128, 1024)

    @property
    def _name(self) -> str:
        return f"PreStackShotGathers3D{self._data_domain.capitalize()}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "ensembleType": "common_source"}

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        for name in self._dim_names:
            self._builder.add_coordinate(name, dimensions=(name,), data_type=ScalarType.INT32)

        # Add non-dimension coordinates
        self._builder.add_coordinate(
            "gun",
            dimensions=("shot_point",),
            data_type=ScalarType.UINT8,
        )
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=("shot_point",),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=("shot_point",),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
        )
        self._builder.add_coordinate(
            "group_coord_x",
            dimensions=("shot_point", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=("shot_point", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
        )
