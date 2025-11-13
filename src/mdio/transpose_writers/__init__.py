"""Views and transformations for MDIO datasets.

This module provides convenience APIs for creating different views and
transformations of MDIO Variables, including repartitioning operations and
sharding capabilities.
"""

from mdio.transpose_writers.chunking import from_variable as chunk_variable
from mdio.transpose_writers.lod import from_variable as lod_variable
from mdio.transpose_writers.shard import from_variable as shard_variable

__all__ = [
    "chunk_variable",
    "lod_variable",
    "shard_variable",
]

