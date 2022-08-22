"""MDIO Data conversion API."""


from .mdio import mdio_to_segy
from .segy import segy_to_mdio


__all__ = ["mdio_to_segy", "segy_to_mdio"]
