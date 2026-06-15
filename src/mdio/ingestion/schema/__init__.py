"""Immutable, format-agnostic schema models describing a dataset before any data is scanned.

These models (``ResolvedSchema``, ``DimensionSpec``) and the ``SchemaEffect`` reshape contract
know nothing about any ingestion format. Concrete effects and the logic that selects them live
with the format that needs them (for SEG-Y grid overrides, see
:mod:`mdio.ingestion.segy.schema_effects`).
"""

from mdio.ingestion.schema.models import DimensionSpec
from mdio.ingestion.schema.models import ResolvedSchema
from mdio.ingestion.schema.models import SchemaEffect

__all__ = [
    "DimensionSpec",
    "ResolvedSchema",
    "SchemaEffect",
]
