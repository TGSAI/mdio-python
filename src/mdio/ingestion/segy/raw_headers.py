"""Experimental raw trace-header ingestion, isolated for one-file removal.

The ``MDIO__IMPORT__RAW_HEADERS`` feature is experimental and expected to change or be
removed. All of its gating and variable construction lives here so that retiring it is a
single-file delete plus removing the call site in :mod:`mdio.ingestion.segy.pipeline`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import zarr

from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.constants import ZarrFormat
from mdio.core.config import MDIOSettings

if TYPE_CHECKING:
    from typing import Any

    from mdio.ingestion.schema import ResolvedSchema

logger = logging.getLogger(__name__)


def build_raw_header_variables(schema: ResolvedSchema) -> list[dict[str, Any]]:
    """Return extra-variable specs for the experimental raw-headers feature.

    Args:
        schema: The resolved schema; its spatial dimensions and chunk shape size the
            ``raw_headers`` variable.

    Returns:
        A single-element list describing the ``raw_headers`` variable when the feature is
        enabled and the active Zarr format supports it (v3); otherwise an empty list.
    """
    settings = MDIOSettings()
    if not settings.raw_headers:
        return []

    if zarr.config.get("default_zarr_format") == ZarrFormat.V2:
        logger.warning("Raw headers are only supported for Zarr v3. Skipping raw headers.")
        return []

    logger.warning("MDIO__IMPORT__RAW_HEADERS is experimental and expected to change or be removed.")
    spatial_dim_names = tuple(dim.name for dim in schema.dimensions if dim.is_spatial)
    chunk_metadata = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=schema.chunk_shape[:-1]))
    return [
        {
            "name": "raw_headers",
            "long_name": "Raw Headers",
            "dimensions": spatial_dim_names,
            "data_type": ScalarType.BYTES240,
            "compressor": Blosc(cname=BloscCname.zstd),
            "coordinates": None,
            "metadata": VariableMetadata(chunk_grid=chunk_metadata),
        }
    ]
