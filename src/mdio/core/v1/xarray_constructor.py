"""Construct an Xarray Dataset from an MDIO v1 Dataset and write to Zarr."""
import xarray as xr
import numpy as np
import dask.array as da
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from typing import Any

from mdio.schema.v1.dataset import Dataset as MDIODataset
from mdio.schema.dimension import NamedDimension
from mdio.schema.dtype import ScalarType, StructuredType


def construct_xarray_dataset(mdio_ds: MDIODataset) -> xr.Dataset:
    """Build an empty xarray.Dataset with correct dimensions and dtypes."""
    # Collect dimension sizes
    dims: dict[str, int] = {}
    for var in mdio_ds.variables:
        for d in var.dimensions:
            if isinstance(d, NamedDimension):
                dims[d.name] = d.size

    # Build data variables
    data_vars: dict[str, xr.DataArray] = {}
    for var in mdio_ds.variables:
        dim_names = [d.name if isinstance(d, NamedDimension) else d for d in var.dimensions]
        shape = tuple(dims[name] for name in dim_names)
        dt = var.data_type
        if isinstance(dt, ScalarType):
            dtype = np.dtype(dt.value)
        elif isinstance(dt, StructuredType):
            dtype = np.dtype([(f.name, f.format.value) for f in dt.fields])
        else:
            raise TypeError(f"Unsupported data_type: {dt}")
        # arr = da.zeros(shape, dtype=dtype)
        arr = np.zeros(shape, dtype=dtype)
        data_array = xr.DataArray(arr, dims=dim_names)
        # set default fill_value to zero instead of NaN
        data_array.encoding['fill_value'] = 0.0  # TODO: This seems to be ignored by xarray
        # attach variable metadata into DataArray attributes, excluding nulls and chunkGrid
        if var.metadata is not None:
            md = var.metadata.model_dump(
                by_alias=True,
                exclude_none=True,
                exclude={"chunk_grid"},
            )
            data_array.attrs.update(md)
        data_vars[var.name] = data_array

    ds = xr.Dataset(data_vars)
    # Attach metadata as attrs
    ds.attrs["apiVersion"] = mdio_ds.metadata.api_version
    ds.attrs["createdOn"] = str(mdio_ds.metadata.created_on)
    if mdio_ds.metadata.attributes:
        ds.attrs["attributes"] = mdio_ds.metadata.attributes
    return ds


def to_mdio_zarr(mdio_ds: MDIODataset, store: str, **kwargs: Any) -> xr.Dataset:
    """Construct an xarray.Dataset and write it to a Zarr store. Returns the xarray.Dataset."""
    ds = construct_xarray_dataset(mdio_ds)
    # Write to Zarr format v2 with consolidated metadata and all attributes
    enc = V2ChunkKeyEncoding(separator="/").to_dict()
    global_encodings = {}

    for var in mdio_ds.variables:
        fill_value = 0
        if isinstance(var.data_type, StructuredType):
            # Create a structured fill value that matches the dtype
            # fill_value = np.zeros(1, dtype=[(f.name, f.format.value) for f in var.data_type.fields])[0]
            # TODO: Re-enable this once xarray supports this PR https://github.com/zarr-developers/zarr-python/pull/3015
            continue
        chunks = None
        if var.metadata is not None and var.metadata.chunk_grid is not None:
            chunks = var.metadata.chunk_grid.configuration.chunk_shape
        global_encodings[var.name] = {
            "chunks": chunks,
            # TODO: Re-enable this once xarray supports this PR https://github.com/pydata/xarray/pull/10274
            # "chunk_key_encoding": enc,
            "_FillValue": fill_value,
            "dtype": var.data_type,
        }

    ds.to_mdio(store, 
               mode="w", 
               zarr_format=2,
               consolidated=True,
               safe_chunks=False,  # This ignores the Dask chunks
               compute=False,  # Ensures only the metadata is written
               encoding=global_encodings,
               **kwargs)
    return ds