"""SeismicPreStackTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain


class Seismic3DPreStackStreamerFieldRecordsTemplate(AbstractDatasetTemplate):
    """Seismic pre-stack time Dataset template.

    A generalized template for pre-stack field records in either 2D or 3D.
        - Common-shot dataset
        - Common-channel dataset

    Args:
        data_domain: The domain of the dataset.
    """

    def __init__(self, data_domain: SeismicDataDomain):
        super().__init__(data_domain=data_domain)

        self._spatial_dim_names = ("shot_line", "gun", "shot_point", "cable", "channel")
        self._dim_names = (*self._spatial_dim_names, self._data_domain)
        self._physical_coord_names = ("source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y")
        self._logical_coord_names = ("orig_field_record_num",)
        # TODO(Anyone): Disable chunking in time domain when support is merged.
        # https://github.com/TGSAI/mdio-python/pull/723
        # self._var_chunk_shape = (1, 1, 16, 1, 32, -1)
        self._var_chunk_shape = (1, 1, 16, 1, 32, 1024)

    @property
    def _name(self) -> str:
        return f"PreStackStreamerFieldRecords3D{self._data_domain.capitalize()}"

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
            "orig_field_record_num",
            dimensions=("shot_line", "gun", "shot_point"),
            data_type=ScalarType.INT32,
        )
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=("shot_line", "gun", "shot_point"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_x")),
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=("shot_line", "gun", "shot_point"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_y")),
        )
        self._builder.add_coordinate(
            "group_coord_x",
            dimensions=("shot_line", "gun", "shot_point", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("group_coord_x")),
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=("shot_line", "gun", "shot_point", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("group_coord_y")),
        )
