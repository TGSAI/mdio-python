"""MDIO library."""

from importlib import metadata

from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.converters import mdio_to_segy
from mdio.converters import segy_to_mdio
from mdio.segy.geometry import GridOverrides

try:
    __version__ = metadata.version("multidimio")
except metadata.PackageNotFoundError:
    __version__ = "unknown"


__all__ = [
    "__version__",
    "open_mdio",
    "to_mdio",
    "mdio_to_segy",
    "segy_to_mdio",
    "GridOverrides",
]
