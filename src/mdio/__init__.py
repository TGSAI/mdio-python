"""MDIO library."""


from importlib import metadata

from mdio.api import MDIOReader
from mdio.api import MDIOWriter
from mdio.api.convenience import copy_mdio
from mdio.converters import mdio_to_segy
from mdio.converters import segy_to_mdio


__all__ = [
    "MDIOReader",
    "MDIOWriter",
    "copy_mdio",
    "mdio_to_segy",
    "segy_to_mdio",
]


try:
    __version__ = metadata.version("multidimio")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
