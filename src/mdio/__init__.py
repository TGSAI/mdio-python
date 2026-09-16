"""MDIO library."""

from __future__ import annotations

from importlib import metadata

from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.converters import mdio_to_segy
from mdio.converters import segy_to_mdio
from mdio.ingestion import ResolvedSchema
from mdio.optimize.access_pattern import OptimizedAccessPatternConfig
from mdio.optimize.access_pattern import optimize_access_patterns
from mdio.segy.geometry import GridOverrides
from mdio.segy.headers import SegyFileHeaders
from mdio.segy.headers import update_segy_file_headers

try:
    __version__ = metadata.version("multidimio")
except metadata.PackageNotFoundError:
    __version__ = "unknown"


__all__ = [
    "__version__",
    "GridOverrides",
    "SegyFileHeaders",
    "open_mdio",
    "to_mdio",
    "update_segy_file_headers",
    "mdio_to_segy",
    "segy_to_mdio",
    "OptimizedAccessPatternConfig",
    "optimize_access_patterns",
    "ResolvedSchema",
]
