"""SeismicPreStackTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain


class SeismicPreStackTemplate(AbstractDatasetTemplate):
    """Seismic pre-stack time Dataset template.

    This should be used for both 2D and 3D datasets. Common-shot or common-channel datasets

    Args:
        domain: The domain of the dataset.
    """

    def __init__(self, data_domain: SeismicDataDomain):
        super().__init__(data_domain=data_domain)

        self._coord_dim_names = [
            "shot_line",
            "gun",
            "shot_point",
            "cable",
            "channel",
        ]  # Custom coordinates for shot gathers
        self._dim_names = [*self._coord_dim_names, self._data_domain]
        self._coord_names = [
            "energy_source_point_number",
            "source_coord_x",
            "source_coord_y",
            "group_coord_x",
            "group_coord_y",
        ]
        self._var_chunk_shape = [1, 1, 16, 1, 32, -1]

    @property
    def _name(self) -> str:
        return f"PreStackGathers3D{self._data_domain.capitalize()}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {
            "surveyDimensionality": "3D",
            "ensembleType": "shot_point",
            "processingStage": "pre-stack",
        }

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        for name in self._dim_names:
            self._builder.add_coordinate(name, dimensions=(name,), data_type=ScalarType.INT32)

        # Add non-dimension coordinates
        self._builder.add_coordinate(
            "energy_source_point_number",
            dimensions=("shot_line", "gun", "shot_point"),
            data_type=ScalarType.INT32,
        )
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=("shot_line", "gun", "shot_point"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=("shot_line", "gun", "shot_point"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
        )
        self._builder.add_coordinate(
            "group_coord_x",
            dimensions=("shot_line", "gun", "shot_point", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=("shot_line", "gun", "shot_point", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
        )
