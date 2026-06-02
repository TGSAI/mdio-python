"""Immutable schema models that describe a dataset before any data is scanned.

``SchemaResolver`` lives in :mod:`mdio.ingestion.schema.resolver` and is imported from there
directly to avoid a circular import (the resolver depends on the SEG-Y index-strategy
registry, which in turn depends on these models).
"""

from mdio.ingestion.schema.models import CollapseToTraceEffect
from mdio.ingestion.schema.models import DimensionSpec
from mdio.ingestion.schema.models import InsertTraceDimEffect
from mdio.ingestion.schema.models import ResolvedSchema
from mdio.ingestion.schema.models import SchemaEffect

__all__ = [
    "CollapseToTraceEffect",
    "DimensionSpec",
    "InsertTraceDimEffect",
    "ResolvedSchema",
    "SchemaEffect",
]
