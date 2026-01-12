"""MDIO library."""

from __future__ import annotations

from importlib import metadata

from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.converters import mdio_to_segy
from mdio.converters import segy_to_mdio
from mdio.optimize.access_pattern import OptimizedAccessPatternConfig
from mdio.optimize.access_pattern import optimize_access_patterns

try:
    __version__ = metadata.version("multidimio")
except metadata.PackageNotFoundError:
    __version__ = "unknown"

# Import numpy_to_mdio after __version__ is set to avoid circular import
from mdio.converters.numpy import numpy_to_mdio

__all__ = [
    "__version__",
    "open_mdio",
    "to_mdio",
    "mdio_to_segy",
    "numpy_to_mdio",
    "segy_to_mdio",
    "OptimizedAccessPatternConfig",
    "optimize_access_patterns",
]
