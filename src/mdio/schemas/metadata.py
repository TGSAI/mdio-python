"""Metadata schemas and conventions."""


from typing import Any

from pydantic import Field

from mdio.schemas.chunk_grid import RectilinearChunkGrid
from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.core import CamelCaseStrictModel


class ChunkGridMetadata(CamelCaseStrictModel):
    """Definition of chunk grid."""

    chunk_grid: RegularChunkGrid | RectilinearChunkGrid | None = Field(
        default=None,
        description="Chunk grid specification for the array.",
    )


class VersionedMetadataConvention(CamelCaseStrictModel):
    """Data model for versioned metadata convention."""

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Override default model dump to use alias."""
        return super().model_dump(*args, **kwargs, by_alias=True)

    def model_dump_json(self, *args, **kwargs) -> str:
        """Override default model dump json to use alias."""
        return super().model_dump_json(*args, **kwargs, by_alias=True)


class UserAttributes(CamelCaseStrictModel):
    """User defined attributes as key/value pairs."""

    attributes: dict[str, Any] | None = Field(
        default=None,
        description="User defined attributes as key/value pairs.",
    )
