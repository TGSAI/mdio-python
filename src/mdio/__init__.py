"""MDIO library."""

from importlib import metadata

try:
    __version__ = metadata.version("multidimio")
except metadata.PackageNotFoundError:
    __version__ = "unknown"

from mdio.api.create import create_empty
from mdio.api.create import create_empty_like
from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.converters.mdio import mdio_to_segy
from mdio.converters.segy import segy_to_mdio

__all__ = [
    "__version__",
    "open_mdio",
    "to_mdio",
    "mdio_to_segy",
    "segy_to_mdio",
    "create_empty",
    "create_empty_like",
]
