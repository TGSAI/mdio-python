"""Seismic3DShotReceiverLineGathersTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import CoordinateSpec
from mdio.builder.templates.types import SeismicDataDomain


class Seismic3DShotReceiverLineGathersTemplate(AbstractDatasetTemplate):
    """Seismic 3D shot-ordered gathers with receiver lines template."""

    def __init__(self, data_domain: SeismicDataDomain = "time"):
        super().__init__(data_domain=data_domain)

        self._spatial_dim_names = ("shot_line", "shot_point", "receiver_line", "receiver")
        self._dim_names = (*self._spatial_dim_names, self._data_domain)
        self._physical_coord_names = (
            "source_coord_x",
            "source_coord_y",
            "group_coord_x",
            "group_coord_y",
        )
        self._logical_coord_names = ("orig_field_record_num",)
        self._var_chunk_shape = (1, 32, 1, 32, 2048)

    @property
    def _name(self) -> str:
        return "ShotReceiverLineGathers3D"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "common_source"}

    def declare_coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        """Declare shot-line- and receiver-line-indexed coordinates for the 3D shot/receiver-line template."""
        source_dims = ("shot_line", "shot_point")
        group_dims = ("receiver_line", "receiver")
        return (
            CoordinateSpec(name="source_coord_x", dimensions=source_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="source_coord_y", dimensions=source_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="group_coord_x", dimensions=group_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="group_coord_y", dimensions=group_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="orig_field_record_num", dimensions=source_dims, dtype=ScalarType.UINT32),
        )

    def declare_dimension_specs(self) -> dict[str, ScalarType]:
        """Declare the data types for each dimension in this template."""
        return {
            "shot_line": ScalarType.UINT32,
            "shot_point": ScalarType.UINT32,
            "receiver_line": ScalarType.UINT32,
            "receiver": ScalarType.UINT32,
            self._data_domain: ScalarType.INT32,
        }

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        self._builder.add_coordinate(
            "shot_line",
            dimensions=("shot_line",),
            data_type=self._dim_dtype("shot_line"),
        )
        self._builder.add_coordinate(
            "shot_point",
            dimensions=("shot_point",),
            data_type=self._dim_dtype("shot_point"),
        )
        self._builder.add_coordinate(
            "receiver_line",
            dimensions=("receiver_line",),
            data_type=self._dim_dtype("receiver_line"),
        )
        self._builder.add_coordinate(
            "receiver",
            dimensions=("receiver",),
            data_type=self._dim_dtype("receiver"),
        )
        self._builder.add_coordinate(
            self._data_domain,
            dimensions=(self._data_domain,),
            data_type=self._dim_dtype(self._data_domain),
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(self._data_domain)),
        )

        # Add non-dimension coordinates
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=("shot_line", "shot_point"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_x")),
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=("shot_line", "shot_point"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_y")),
        )
        self._builder.add_coordinate(
            "group_coord_x",
            dimensions=("receiver_line", "receiver"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("group_coord_x")),
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=("receiver_line", "receiver"),
            data_type=ScalarType.FLOAT64,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("group_coord_y")),
        )
        self._builder.add_coordinate(
            "orig_field_record_num",
            dimensions=("shot_line", "shot_point"),
            data_type=ScalarType.UINT32,
        )
