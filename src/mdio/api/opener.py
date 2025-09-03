"""Utils for reading MDIO dataset."""

import xarray as xr

from mdio.core.storage_location import StorageLocation


def open_dataset(storage_location: StorageLocation) -> xr.Dataset:
    """Open a Zarr dataset from the specified storage location."""
    # NOTE: If mask_and_scale is not set,
    # Xarray will convert int to float and replace _FillValue with NaN
    # We are using chunks={} to force Xarray to create Dask arrays
    return xr.open_dataset(storage_location.uri, engine="zarr", chunks={}, mask_and_scale=False)
