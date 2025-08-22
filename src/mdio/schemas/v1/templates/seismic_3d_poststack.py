"""Seismic3DPostStackTemplate MDIO v1 dataset templates."""

from mdio.schemas import ScalarType
from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.chunk_grid import RegularChunkShape
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate


class Seismic3DPostStackTemplate(AbstractDatasetTemplate):
    """Seismic post-stack 3D time or depth Dataset template.

    Access Patterns:
        inline: full lines
        crossline: full lines
        depth/time: full slices or horizon slices (attributes)
    """

    def __init__(self, domain: str):
        super().__init__(domain=domain)

        self._coord_dim_names = ["inline", "crossline"]
        self._dim_names = [*self._coord_dim_names, self._trace_domain]
        self._coord_names = ["cdp_x", "cdp_y"]
        self._var_chunk_shape = (128, 128, 128)

    @property
    def _name(self) -> str:
        return f"PostStack3D{self._trace_domain.capitalize()}"

    def _load_dataset_attributes(self) -> UserAttributes:
        return UserAttributes(
            attributes={
                "surveyDimensionality": "3D",
                "ensembleType": "line",
                "processingStage": "post-stack",
            }
        )

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        for name in self._dim_names:
            self._builder.add_coordinate(
                name,
                dimensions=[name],
                data_type=ScalarType.INT32,
                metadata_info=None,
            )

        # Add non-dimension coordinates
        coord_chunk_shape = (384, 384)  # 1.1MB with fp64
        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=coord_chunk_shape))
        chunk_meta = ChunkGridMetadata(chunk_grid=chunk_grid)
        self._builder.add_coordinate(
            "cdp_x",
            dimensions=["inline", "crossline"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit, chunk_meta],
        )
        self._builder.add_coordinate(
            "cdp_y",
            dimensions=["inline", "crossline"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit, chunk_meta],
        )
