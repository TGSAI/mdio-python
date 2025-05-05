"""
Overloads for xarray.

The intent of overloading here is:
1. To provide a consistent mdio.* naming scheme.
2. To simplify the API for users where it makes sense (e.g. MDIO v1 uses Zarr and not HDF5).
"""
import xarray as xr
from xarray import Dataset as _Dataset, DataArray as _DataArray


class MDIODataset(_Dataset):
    """xarray.Dataset subclass with MDIO v1 extensions."""
    __slots__ = ()

    def to_mdio(self, store=None, *args, **kwargs):
        """
        Alias for `.to_zarr()`, prints a greeting, and writes to Zarr store.
        """
        print("👋 hello world from mdio.to_mdio!")
        return super().to_zarr(store=store, *args, **kwargs)


class MDIODataArray(_DataArray):
    """xarray.DataArray subclass with MDIO v1 extensions."""
    __slots__ = ()

    def to_mdio(self, store=None, *args, **kwargs):
        """
        Alias for `.to_zarr()`, prints a greeting, and writes to Zarr store.
        """
        print("👋 hello world from mdio.to_mdio!")
        return super().to_zarr(store=store, *args, **kwargs)


class MDIO:
    """MDIO namespace for overloaded types and functions."""
    Dataset = MDIODataset
    DataArray = MDIODataArray

    @staticmethod
    def open(store, *args, engine="zarr", consolidated=False, **kwargs):
        """
        Open a Zarr store as an MDIODataset. Prints a greeting and casts
        the returned xarray.Dataset (and its variables) to the MDIO subclasses.
        """
        print("👋 hello world from mdio.open!")
        ds = xr.open_dataset(
            store,
            engine=engine,
            consolidated=consolidated,
            *args,
            **kwargs,
        )
        # Cast Dataset to MDIODataset
        ds.__class__ = MDIODataset
        # Cast each DataArray in data_vars and coords
        for name, var in ds.data_vars.items():
            var.__class__ = MDIODataArray
        for name, coord in ds.coords.items():
            coord.__class__ = MDIODataArray
        return ds


# Create module-level MDIO namespace
mdio = MDIO()
