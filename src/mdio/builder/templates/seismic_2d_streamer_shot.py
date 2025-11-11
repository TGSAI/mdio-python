"""Seismic2DStreamerShotGathersTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas import compressors
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain


class Seismic2DStreamerShotGathersTemplate(AbstractDatasetTemplate):
    """Seismic Shot pre-stack 2D time or depth Dataset template."""

    def __init__(self, data_domain: SeismicDataDomain = "time"):
        super().__init__(data_domain=data_domain)

        self._dim_names = ("shot_point", "channel", self._data_domain)
        self._physical_coord_names = ("source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y")
        self._var_chunk_shape = (16, 32, 2048)

    @property
    def _name(self) -> str:
        return "StreamerShotGathers2D"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "2D", "gatherType": "common_source"}

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        for name in self._dim_names:
            self._builder.add_coordinate(
                name,
                dimensions=(name,),
                data_type=ScalarType.INT32,
                metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(name)),
            )

        # Add non-dimension coordinates
        compressor = compressors.Blosc(cname=compressors.BloscCname.zstd)
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
            dimensions=("shot_point", "channel"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("group_coord_x")),
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=("shot_point", "channel"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("group_coord_y")),
        )
