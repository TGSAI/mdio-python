"""Seismic3DReceiverGathersTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas import compressors
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import CoordinateSpec
from mdio.builder.templates.types import DimCoordinateTypes


class Seismic3DReceiverGathersTemplate(AbstractDatasetTemplate):
    """Seismic 3D receiver gathers template."""

    def __init__(self) -> None:
        super().__init__(data_domain="time")

        self._dim_names = ("receiver", "shot_line", "shot_point", "time")
        self._physical_coord_names = (
            "receiver_x",
            "receiver_y",
            "source_coord_x",
            "source_coord_y",
        )
        self._logical_coord_names = ()
        self._var_chunk_shape = (1, 1, 512, 4096)

    @property
    def _name(self) -> str:
        return "ReceiverGathers3D"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "receiver_gathers"}

    def declare_coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        """Declare receiver- and shot-indexed coordinates for the 3D receiver gathers template."""
        receiver_dim = ("receiver",)
        shot_dims = ("shot_line", "shot_point")
        return (
            CoordinateSpec(name="receiver_x", dimensions=receiver_dim, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="receiver_y", dimensions=receiver_dim, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="source_coord_x", dimensions=shot_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="source_coord_y", dimensions=shot_dims, dtype=ScalarType.FLOAT64),
        )

    def declare_dim_coordinate_types(self) -> DimCoordinateTypes:
        """Declare the data types for each dimension coordinate in this template."""
        return {
            "receiver": ScalarType.UINT32,
            "shot_line": ScalarType.UINT32,
            "shot_point": ScalarType.UINT32,
            self._data_domain: ScalarType.INT32,
        }

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        self._builder.add_coordinate(
            "receiver",
            dimensions=("receiver",),
            data_type=self._dim_dtype("receiver"),
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("receiver")),
        )
        self._builder.add_coordinate(
            "shot_line",
            dimensions=("shot_line",),
            data_type=self._dim_dtype("shot_line"),
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("shot_line")),
        )
        self._builder.add_coordinate(
            "shot_point",
            dimensions=("shot_point",),
            data_type=self._dim_dtype("shot_point"),
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("shot_point")),
        )
        self._builder.add_coordinate(
            self.trace_domain,
            dimensions=(self.trace_domain,),
            data_type=self._dim_dtype(self.trace_domain),
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(self.trace_domain)),
        )

        # Add non-dimension coordinates
        compressor = compressors.Blosc(cname=compressors.BloscCname.zstd)

        # Receiver coordinates (fixed per receiver)
        self._builder.add_coordinate(
            "receiver_x",
            dimensions=("receiver",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("receiver_x")),
        )
        self._builder.add_coordinate(
            "receiver_y",
            dimensions=("receiver",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("receiver_y")),
        )

        # Source coordinates (vary by shot_line and shot_point)
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=("shot_line", "shot_point"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_x")),
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=("shot_line", "shot_point"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_y")),
        )
