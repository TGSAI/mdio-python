"""MDIO library."""

from importlib import metadata

from mdio.api import MDIOReader
from mdio.api import MDIOWriter
from mdio.api.convenience import copy_mdio
from mdio.api.opener import open_dataset
from mdio.converters import mdio_to_segy
from mdio.converters import numpy_to_mdio
from mdio.converters import segy_to_mdio
from mdio.core.dimension import Dimension
from mdio.core.factory import MDIOCreateConfig
from mdio.core.factory import MDIOVariableConfig
from mdio.core.factory import create_empty
from mdio.core.factory import create_empty_like
from mdio.core.grid import Grid
from mdio.core.storage_location import StorageLocation

__all__ = [
    "MDIOReader",
    "MDIOWriter",
    "copy_mdio",
    "open_dataset",
    "mdio_to_segy",
    "numpy_to_mdio",
    "segy_to_mdio",
    "Dimension",
    "MDIOCreateConfig",
    "MDIOVariableConfig",
    "create_empty",
    "create_empty_like",
    "Grid",
    "StorageLocation",
]


try:
    __version__ = metadata.version("multidimio")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
