"""Seismic2DPostStackTemplate MDIO v1 dataset templates."""

from mdio.schemas import ScalarType
from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.chunk_grid import RegularChunkShape
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate


class Seismic2DPostStackTemplate(AbstractDatasetTemplate):
    """Seismic post-stack 2D time or depth Dataset template.

    Access Patterns:
        cdp: full lines
    """

    def __init__(self, domain: str):
        super().__init__(domain=domain)

        self._coord_dim_names = ["cdp"]
        self._dim_names = [*self._coord_dim_names, self._trace_domain]
        self._coord_names = ["cdp_x", "cdp_y"]
        self._var_chunk_shape = (1024, 1024)

    @property
    def _name(self) -> str:
        return f"PostStack2D{self._trace_domain.capitalize()}"

    def _load_dataset_attributes(self) -> UserAttributes:
        return UserAttributes(
            attributes={
                "surveyDimensionality": "2D",
                "ensembleType": "line",
                "processingStage": "post-stack",
                # TODO: Consider other attributes from text header
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
        coord_chunk_shape = (131072,)  # 1MB with fp64
        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=coord_chunk_shape))
        chunk_meta = ChunkGridMetadata(chunk_grid=chunk_grid)
        self._builder.add_coordinate(
            "cdp_x",
            dimensions=["cdp"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit, chunk_meta],
        )
        self._builder.add_coordinate(
            "cdp_y",
            dimensions=["cdp"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit, chunk_meta],
        )
