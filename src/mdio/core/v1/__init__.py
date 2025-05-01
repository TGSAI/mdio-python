# mdio/__init__.py

import xarray as _xr
from ._overloads import open_mdio, to_mdio

__all__ = [
    # explicit overrides / aliases
    "open_mdio",
    "to_mdio",
    # everything else will be auto-populated by __dir__ / __getattr__
]

def __getattr__(name: str):
    """
    Fallback: anything not defined in mdio/__init__.py
    gets looked up on xarray.
    """
    if hasattr(_xr, name):
        return getattr(_xr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    """
    Make dir(mdio) list our overrides and then all public xarray names.
    """
    xr_public = [n for n in dir(_xr) if not n.startswith("_")]
    return sorted(__all__ + xr_public)
