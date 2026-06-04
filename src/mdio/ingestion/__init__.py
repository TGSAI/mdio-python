"""MDIO ingestion pipeline components.

This is the advanced ingestion namespace. ``CoordinateSpec`` is intentionally not
re-exported here; its canonical home is :mod:`mdio.builder.templates.types`.
"""

from mdio.ingestion.schema import DimensionSpec
from mdio.ingestion.schema import ResolvedSchema
from mdio.ingestion.segy.index_strategies import IndexStrategy
from mdio.ingestion.segy.index_strategies import IndexStrategyRegistry
from mdio.ingestion.segy.pipeline import segy_to_mdio

__all__ = [
    "DimensionSpec",
    "IndexStrategy",
    "IndexStrategyRegistry",
    "ResolvedSchema",
    "segy_to_mdio",
]
