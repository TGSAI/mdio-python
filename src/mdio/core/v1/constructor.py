"""Construct an Xarray Dataset from an MDIO v1 Dataset and write to Zarr."""
import xarray as xr
import numpy as np
import dask.array as da
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from typing import Any

from mdio.schema.v1.dataset import Dataset as MDIODataset
from mdio.schema.dimension import NamedDimension
from mdio.schema.dtype import ScalarType, StructuredType
from mdio.schema.compressors import Blosc, ZFP
from mdio.schema.v1.variable import Coordinate


from numcodecs import Blosc as NumcodecsBlosc

try:
    import zfpy as BaseZFPY # Baser library
    from numcodecs import ZFPY as NumcodecsZFPY  # Codec
except ImportError:
    print(f"Tried to import zfpy and numcodes zfpy but failed because {ImportError}")
    BaseZFPY = None
    NumcodecsZFPY = None

def _convert_compressor(model: Blosc | ZFP | None) -> NumcodecsBlosc | NumcodecsZFPY | None:
    if isinstance(model, Blosc):
        return NumcodecsBlosc(
            cname=model.algorithm.value,
            clevel=model.level,
            shuffle=model.shuffle.value,
            blocksize=model.blocksize if model.blocksize > 0 else 0
        )
    elif isinstance(model, ZFP):
        if BaseZFPY is None or NumcodecsZFPY is None:
            raise ImportError("zfpy and numcodecs are required to use ZFP compression")
        return NumcodecsZFPY(
            mode=model.mode.value,
            tolerance=model.tolerance,
            rate=model.rate,
            precision=model.precision,
        )
    elif model is None:
        return None
    else:
        raise TypeError(f"Unsupported compressor model: {type(model)}")


def _construct_mdio_dataset(mdio_ds: MDIODataset) -> xr.Dataset:
    """Build an MDIO dataset with correct dimensions and dtypes.
    
    This internal function constructs the underlying data structure for an MDIO dataset,
    handling dimension mapping, data types, and metadata organization.
    
    Args:
        mdio_ds: The source MDIO dataset to construct from.
        
    Returns:
        The constructed dataset with proper MDIO structure and metadata.
    """
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
        # TODO: This seems to be ignored by xarray.
        # Setting in the _generate_encodings() function does work though.
        data_array.encoding['fill_value'] = 0.0
        
        # Set long_name if present
        if var.long_name is not None:
            data_array.attrs["long_name"] = var.long_name
            
        # Set coordinates if present, excluding dimension names
        if var.coordinates is not None:
            # Get the set of dimension names for this variable
            dim_set = set(dim_names)
            # Filter out any coordinates that are also dimensions
            coord_names = [
                c.name if isinstance(c, Coordinate) else c 
                for c in var.coordinates 
                if (c.name if isinstance(c, Coordinate) else c) not in dim_set
            ]
            if coord_names:  # Only set coordinates if there are any non-dimension coordinates
                data_array.attrs["coordinates"] = " ".join(coord_names)
            
        # attach variable metadata into DataArray attributes, excluding nulls and chunkGrid
        if var.metadata is not None:
            md = var.metadata.model_dump(
                by_alias=True,
                exclude_none=True,
                exclude={"chunk_grid"},
            )
            # Convert single-element lists to objects
            for key, value in md.items():
                if isinstance(value, list) and len(value) == 1:
                    md[key] = value[0]
            data_array.attrs.update(md)
        data_vars[var.name] = data_array

    ds = xr.Dataset(data_vars)
    # Attach metadata as attrs
    ds.attrs["apiVersion"] = mdio_ds.metadata.api_version
    ds.attrs["createdOn"] = str(mdio_ds.metadata.created_on)
    ds.attrs["name"] = mdio_ds.metadata.name
    if mdio_ds.metadata.attributes:
        ds.attrs["attributes"] = mdio_ds.metadata.attributes
    return ds




def Write_MDIO_metadata(mdio_ds: MDIODataset, store: str, **kwargs: Any) -> xr.Dataset:
    """Write MDIO metadata to a Zarr store and return the constructed xarray.Dataset.
    
    This function constructs an xarray.Dataset from the MDIO dataset and writes its metadata
    to a Zarr store. The actual data is not written, only the metadata structure is created.
    """
    ds = _construct_mdio_dataset(mdio_ds)
    # Write to Zarr format v2 with consolidated metadata and all attributes
    
    def _generate_encodings() -> dict:
        """Generate encodings for each variable in the MDIO dataset.
        
        Returns:
            Dictionary mapping variable names to their encoding configurations.
        """
        dimension_separator_encoding = V2ChunkKeyEncoding(separator="/").to_dict()
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
                # "chunk_key_encoding": dimension_separator_encoding,
                "_FillValue": fill_value,
                "dtype": var.data_type,
                "compressors": _convert_compressor(var.compressor),
            }
        return global_encodings

    ds.to_mdio(store, 
               mode="w", 
               zarr_format=2,
               consolidated=True,
               safe_chunks=False,  # This ignores the Dask chunks
               compute=False,  # Ensures only the metadata is written
               encoding=_generate_encodings(),
               **kwargs)
    return ds