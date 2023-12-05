"""Unit schemas specific to MDIO v1."""


from pydantic import Field

from mdio.schemas.chunk_grid import RectilinearChunkGrid
from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.core import StrictCamelBaseModel


class ChunkGridMetadata(StrictCamelBaseModel):
    """Definition of chunk grid."""

    chunk_grid: RegularChunkGrid | RectilinearChunkGrid | None = Field(
        default=None,
        description="Chunk grid specification for the array.",
    )
