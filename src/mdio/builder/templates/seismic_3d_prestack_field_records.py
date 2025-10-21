"""SeismicPreStackTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain


class Seismic3DPreStackFieldRecordsTemplate(AbstractDatasetTemplate):
    """Seismic pre-stack time Dataset template.

    This should be used for both 2D and 3D datasets. Common-shot or common-channel datasets

    Args:
        data_domain: The domain of the dataset.
    """

    def __init__(self, data_domain: SeismicDataDomain):
        super().__init__(data_domain=data_domain)

        self._spatial_dim_names = ("shot_line", "gun", "shot_point", "cable", "channel")
        self._dim_names = (*self._spatial_dim_names, self._data_domain)
        self._physical_coord_names = ("source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y")
        self._logical_coord_names = ("orig_field_record_num",)
        # TODO(Dmitriy Repin): Allow specifying full-dimension-extent chunk size in templates.
        # https://github.com/TGSAI/mdio-python/issues/720
        # When implemented, the following will be requesting the chunk size of the last dimension
        # to be equal to the size of the dimension.
        # self._var_chunk_shape = (1, 1, 16, 1, 32, -1)
        # For now, we are hardcoding the chunk size to 1024.
        self._var_chunk_shape = (1, 1, 16, 1, 32, 1024)

    @property
    def _name(self) -> str:
        return f"PreStackFieldRecords3D{self._data_domain.capitalize()}"

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
