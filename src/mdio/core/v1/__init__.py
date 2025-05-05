"""
MDIO core v1 package initialization.
Exposes the MDIO overloads and core v1 functionality.
"""

from ._overloads import mdio
from .constructor import Write_MDIO_metadata

__all__ = [
    "mdio",
    "Write_MDIO_metadata",
]
