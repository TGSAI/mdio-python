"""Seismic3DStreamerFieldRecordsTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain


class Seismic3DStreamerFieldRecordsTemplate(AbstractDatasetTemplate):
    """Seismic 3D streamer shot field records template.

    A generalized template for streamer field records that are optimized for:
        - Common-shot access
        - Common-channel access

    It can also store all the shot-lines of a survey in one MDIO if needed.

    Args:
        data_domain: The domain of the dataset.
    """

    def __init__(self, data_domain: SeismicDataDomain = "time"):
        super().__init__(data_domain=data_domain)

        self._spatial_dim_names = ("sail_line", "gun", "shot_index", "cable", "channel")
        self._calculated_dims = ("shot_index",)
        self._dim_names = (*self._spatial_dim_names, self._data_domain)
        self._physical_coord_names = ("source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y")
        self._logical_coord_names = ("shot_point", "orig_field_record_num")  # ffid
        self._var_chunk_shape = (1, 1, 16, 1, 32, 1024)

    @property
    def _name(self) -> str:
        return "StreamerFieldRecords3D"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyDimensionality": "3D", "gatherType": "common_source"}

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        # EXCLUDE: `shot_index` since its 0-N
        self._builder.add_coordinate(
            "sail_line",
            dimensions=("sail_line",),
            data_type=ScalarType.UINT32,
        )
        self._builder.add_coordinate(
            "gun",
            dimensions=("gun",),
            data_type=ScalarType.UINT8,
        )
        self._builder.add_coordinate(
            "cable",
            dimensions=("cable",),
            data_type=ScalarType.UINT8,
        )
        self._builder.add_coordinate(
            "channel",
            dimensions=("channel",),
            data_type=ScalarType.UINT16,
        )
        self._builder.add_coordinate(
            self._data_domain,
            dimensions=(self._data_domain,),
            data_type=ScalarType.INT32,
        )

        # Add non-dimension coordinates
        self._builder.add_coordinate(
            "orig_field_record_num",
            dimensions=("sail_line", "gun", "shot_index"),
            data_type=ScalarType.UINT32,
        )
        self._builder.add_coordinate(
            "shot_point",
            dimensions=("sail_line", "gun", "shot_index"),
            data_type=ScalarType.UINT32,
        )
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=("sail_line", "gun", "shot_index"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_x")),
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=("sail_line", "gun", "shot_index"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_y")),
        )
        self._builder.add_coordinate(
            "group_coord_x",
            dimensions=("sail_line", "gun", "shot_index", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("group_coord_x")),
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=("sail_line", "gun", "shot_index", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("group_coord_y")),
        )
