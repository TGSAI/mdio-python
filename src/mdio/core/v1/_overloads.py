"""
Overloads for xarray.

The intent of overloading here is:
1. To provide a consistent mdio.* naming scheme.
2. To simplify the API for users where it makes sense (e.g. MDIO v1 uses Zarr and not HDF5).
"""


import xarray as _xr
from xarray import Dataset as _Dataset, DataArray as _DataArray

def open_mdio(store, *args, engine="zarr", consolidated=False, **kwargs):
    """
    Our mdio version of xr.open_zarr. Prints a greeting,
    then calls xr.open_dataset(..., engine="zarr").
    """
    print("👋 hello world from mdio.open_mdio!")
    return _xr.open_dataset(store, *args,
                             engine=engine,
                             consolidated=consolidated,
                             **kwargs)

def to_mdio(self, *args, **kwargs):
    """
    Alias for .to_zarr, renamed to .to_mdio,
    so you get a consistent mdio.* naming.
    """
    print("👋 hello world from mdio.to_mdio!")
    print(f"kwargs: {kwargs}")
    return self.to_zarr(*args, **kwargs)

# Monkey-patch Dataset and DataArray so that you can do:
#   ds.to_mdio(...)        and     arr.to_mdio(...)
_Dataset.to_mdio = to_mdio
_DataArray.to_mdio = to_mdio
