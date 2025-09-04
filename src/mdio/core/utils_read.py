"""Utils for reading MDIO dataset."""

from xarray import Dataset as xr_Dataset
from xarray import open_dataset as xr_open_dataset

from mdio.core.storage_location import StorageLocation


def open_zarr_dataset(storage_location: StorageLocation) -> xr_Dataset:
    """Open a Zarr dataset from the specified storage location."""
    # NOTE: If mask_and_scale is not set,
    # Xarray will convert int to float and replace _FillValue with NaN
    # We are using chunks={} to force Xarray to create Dask arrays
    return xr_open_dataset(storage_location.uri, engine="zarr", chunks={}, mask_and_scale=False)
