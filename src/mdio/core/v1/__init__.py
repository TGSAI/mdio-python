"""MDIO core v1 package initialization.

Exposes the MDIO overloads and core v1 functionality.
"""

from ._overloads import mdio
from .constructor import write_mdio_metadata


__all__ = [
    "mdio",
    "write_mdio_metadata",
]
