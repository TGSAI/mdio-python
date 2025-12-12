"""Seismic2DStreamerShotGathersTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas import compressors
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain
from mdio.core.utils_write import MAX_COORDINATES_BYTES
from mdio.core.utils_write import get_constrained_chunksize


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
                metadata=VariableMetadata(units_v1=self.get_unit_by_key(name)),
            )

        # Add non-dimension coordinates with computed chunk sizes
        # For 1D coordinates (over shot_point only)
        coord_spatial_shape_1d = (self._dim_sizes[0],)  # shot_point only
        coord_chunk_shape_1d = get_constrained_chunksize(
            coord_spatial_shape_1d,
            ScalarType.FLOAT64,
            MAX_COORDINATES_BYTES,
        )
        chunk_grid_1d = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=coord_chunk_shape_1d))

        # For 2D coordinates (over shot_point, channel)
        coord_spatial_shape_2d = (self._dim_sizes[0], self._dim_sizes[1])  # shot_point, channel
        coord_chunk_shape_2d = get_constrained_chunksize(
            coord_spatial_shape_2d,
            ScalarType.FLOAT64,
            MAX_COORDINATES_BYTES,
        )
        chunk_grid_2d = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=coord_chunk_shape_2d))

        compressor = compressors.Blosc(cname=compressors.BloscCname.zstd)
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=("shot_point",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("source_coord_x"), chunk_grid=chunk_grid_1d),
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=("shot_point",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("source_coord_y"), chunk_grid=chunk_grid_1d),
        )
        self._builder.add_coordinate(
            "group_coord_x",
            dimensions=("shot_point", "channel"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("group_coord_x"), chunk_grid=chunk_grid_2d),
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=("shot_point", "channel"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("group_coord_y"), chunk_grid=chunk_grid_2d),
        )
