"""Utils for reading MDIO dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from xarray.core.types import T_Chunks

    from mdio.core.storage_location import StorageLocation


def open_dataset(storage_location: StorageLocation, chunks: T_Chunks = None) -> xr.Dataset:
    """Open a Zarr dataset from the specified storage location.

    Args:
        storage_location: StorageLocation for the dataset.
        chunks: If provided, loads data into dask arrays with new chunking.
            - ``chunks="auto"`` will use dask ``auto`` chunking.
            - ``chunks=None`` loads the data with dask using on disk chunk size.
            - ``chunks=-1`` loads the data with dask using a single chunk for all arrays.

            See Xarray's open_dataset for more details.

    Returns:
        An Xarray dataset opened from the storage location.
    """
    if chunks is None:
        chunks = {}

    # NOTE: If mask_and_scale is not set,
    # Xarray will convert int to float and replace _FillValue with NaN
    # We are using chunks={} to force Xarray to create Dask arrays
    return xr.open_dataset(storage_location.uri, engine="zarr", chunks=chunks, mask_and_scale=False)
