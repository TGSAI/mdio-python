"""Overloads for xarray.

The intent of overloading here is:
1. To provide a consistent mdio.* naming scheme.
2. To simplify the API for users where it makes sense (e.g. MDIO v1 uses Zarr and not HDF5).
"""

from collections.abc import Mapping

import xarray as xr
from xarray import DataArray as _DataArray
from xarray import Dataset as _Dataset


class MDIODataset(_Dataset):
    """xarray.Dataset subclass with MDIO v1 extensions."""

    __slots__ = ()

    def to_mdio(
        self,
        store: str | None = None,
        *args: str | int | float | bool,
        **kwargs: Mapping[str, str | int | float | bool],
    ) -> None:
        """Alias for `.to_zarr()`, prints a greeting, and writes to Zarr store."""
        print("👋 hello world from mdio.to_mdio!")
        return super().to_zarr(*args, store=store, **kwargs)


class MDIODataArray(_DataArray):
    """xarray.DataArray subclass with MDIO v1 extensions."""

    __slots__ = ()

    def to_mdio(
        self,
        store: str | None = None,
        *args: str | int | float | bool,
        **kwargs: Mapping[str, str | int | float | bool],
    ) -> None:
        """Alias for `.to_zarr()`, prints a greeting, and writes to Zarr store."""
        print("👋 hello world from mdio.to_mdio!")
        return super().to_zarr(*args, store=store, **kwargs)


class MDIO:
    """MDIO namespace for overloaded types and functions."""

    Dataset = MDIODataset
    DataArray = MDIODataArray

    @staticmethod
    def open(
        store: str,
        *args: str | int | float | bool,
        engine: str = "zarr",
        consolidated: bool = False,
        **kwargs: Mapping[str, str | int | float | bool],
    ) -> MDIODataset:
        """Open a Zarr store as an MDIODataset.

        Casts the returned xarray.Dataset (and its variables) to the MDIO subclasses.
        """
        print("👋 hello world from mdio.open!")
        ds = xr.open_dataset(
            store,
            *args,
            engine=engine,
            consolidated=consolidated,
            **kwargs,
        )
        # Cast Dataset to MDIODataset
        ds.__class__ = MDIODataset
        # Cast each DataArray in data_vars and coords
        for _name, var in ds.data_vars.values():
            var.__class__ = MDIODataArray
        for _name, coord in ds.coords.values():
            coord.__class__ = MDIODataArray
        return ds


# Create module-level MDIO namespace
mdio = MDIO()
