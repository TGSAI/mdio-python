"""Seismic3DStreamerShotGathersTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas import compressors
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import CoordinateSpec
from mdio.builder.templates.types import SeismicDataDomain


class Seismic3DStreamerShotGathersTemplate(AbstractDatasetTemplate):
    """Seismic Shot pre-stack 3D time or depth Dataset template."""

    def __init__(self, data_domain: SeismicDataDomain = "time"):
        super().__init__(data_domain=data_domain)

        self._dim_names = ("shot_point", "cable", "channel", self._data_domain)
        self._physical_coord_names = ("source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y")
        self._logical_coord_names = ("gun",)
        self._var_chunk_shape = (8, 1, 128, 2048)

    @property
    def _name(self) -> str:
        return "StreamerShotGathers3D"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "common_source"}

    def declare_coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        """Declare shot- and receiver-indexed coordinates for the 3D streamer shot gathers template."""
        shot_dim = ("shot_point",)
        receiver_dims = ("shot_point", "cable", "channel")
        return (
            CoordinateSpec(name="gun", dimensions=shot_dim, dtype=ScalarType.UINT8),
            CoordinateSpec(name="source_coord_x", dimensions=shot_dim, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="source_coord_y", dimensions=shot_dim, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="group_coord_x", dimensions=receiver_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="group_coord_y", dimensions=receiver_dims, dtype=ScalarType.FLOAT64),
        )

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        for name in self._dim_names:
            self._builder.add_coordinate(
                name,
                dimensions=(name,),
                data_type=self._dim_dtype(name),
                metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(name)),
            )

        # Add non-dimension coordinates
        compressor = compressors.Blosc(cname=compressors.BloscCname.zstd)
        self._builder.add_coordinate(
            "gun",
            dimensions=("shot_point",),
            data_type=ScalarType.UINT8,
            compressor=compressor,
        )
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=("shot_point",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_x")),
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=("shot_point",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_y")),
        )
        self._builder.add_coordinate(
            "group_coord_x",
            dimensions=("shot_point", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("group_coord_x")),
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=("shot_point", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("group_coord_y")),
        )
