"""SEG-Y specific implementation module."""

from mdio.segy.headers import SegyFileHeaders
from mdio.segy.headers import update_segy_file_headers

__all__ = [
    "SegyFileHeaders",
    "update_segy_file_headers",
]
