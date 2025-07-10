from typing import Mapping
import warnings

import dask
import xarray as xr
import numpy as np
import zarr

from mdio.schemas.dimension import NamedDimension
from mdio.schemas.dtype import ScalarType, StructuredType
from mdio.schemas.v1 import dataset
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset_builder import _to_dictionary
from mdio.schemas.v1.variable import Coordinate
from mdio.schemas.v1.variable import Variable


def _get_all_named_dimensions(dataset: Dataset) -> dict[str, NamedDimension]:
    all_named_dims: dict[str, NamedDimension] = {}
    for v in dataset.variables:
        if v.dimensions is not None:
            for d in v.dimensions:
                if isinstance(d, NamedDimension):
                    all_named_dims[d.name] = d
                else:
                    # Skip, if this is a named reference
                    # This should not be ever a case for the dataset generated with the dataset builder
                    warnings.warn(
                        f"Unsupported dimension type: {type(d)} in variable {v.name}. "
                        "Expected NamedDimension."
                    )
                    pass
    return all_named_dims

def _get_all_coordinates(dataset: Dataset) -> dict[str, Coordinate]:
    all_coords: dict[str, Coordinate] = {}
    for v in dataset.variables:
        if v.coordinates is not None:
            for c in v.coordinates:
                if isinstance(c, Coordinate) and c.name not in all_coords:
                    all_coords[c.name] = c
    return all_coords

def _get_dimension_names(var: Variable) -> list[str]:
    dim_names: list[str] = []
    if var.dimensions is not None:
        for d in var.dimensions:
            if isinstance(d, NamedDimension):
                dim_names.append(d.name)
            elif isinstance(d, str):
                # This should not be ever a case for the dataset generated with the dataset builder
                warnings.warn(f"Unsupported dimension type: 'str' in variable {var.name}. "
                        "Expected NamedDimension."
                    )
                dim_names.append(d)
            else:
                err = f"Unsupported dimension type: {type(d)} in variable {var.name}"
                raise TypeError(err)
    return dim_names
 
def _get_coord_names(var: Variable) -> list[str]:
    coord_names: list[str] = []
    if var.coordinates is not None:
        for c in var.coordinates:
            if isinstance(c, Coordinate):
                coord_names.append(c.name)
            elif isinstance(c, str):
                coord_names.append(c)
            else:
                err = f"Unsupported coordinate type: {type(c)} in variable {var.name}"
                raise TypeError(err)
    return coord_names

def _get_np_datatype(var: Variable) -> np.dtype:
    data_type = var.data_type
    if isinstance(data_type, ScalarType):
        return np.dtype(data_type.value)
    elif isinstance(data_type, StructuredType):
        return np.dtype([(f.name, f.format.value) for f in data_type.fields])
    else:
        raise TypeError(f"Unsupported data_type: {data_type}")

def _get_zarr_shape(var: Variable) -> tuple[int, ...]:
    # NOTE: This assumes that the variable dimensions are all NamedDimension
    return tuple(dim.size for dim in var.dimensions)

def _get_zarr_chunks(var: Variable) -> tuple[int, ...]:
    """Get the chunk shape for a variable, defaulting to its shape if no chunk grid is defined."""
    if var.metadata is not None and var.metadata.chunk_grid is not None:
        return var.metadata.chunk_grid.configuration.chunk_shape
    else:
        # Default to full shape if no chunk grid is defined
        return _get_zarr_shape(var)

def to_xarray_dataset(ds: Dataset) -> xr.DataArray:  # noqa: PLR0912
    """Build an MDIO dataset with correct dimensions and dtypes.

    This internal function constructs the underlying data structure for an MDIO dataset,
    handling dimension mapping, data types, and metadata organization.

    Args:
        mdio_ds: The source MDIO dataset to construct from.

    Returns:
        The constructed dataset with proper MDIO structure and metadata.

    Raises:
        TypeError: If an unsupported data type is encountered.
    """

    # See the xarray tutorial for more details on how to create datasets:
    # https://tutorial.xarray.dev/fundamentals/01.1_creating_data_structures.html   

    # all_dims = _get_all_named_dimensions(ds)
    # all_coords = _get_all_coordinates(ds)

    # Build all variables
    data_arrays: dict[str, xr.DataArray] = {}
    for v in ds.variables:

        # Use dask array instead of numpy array for lazy evaluation
        shape = _get_zarr_shape(v)
        dtype = _get_np_datatype(v)
        chunks = _get_zarr_chunks(v)
        arr = dask.array.zeros(shape, dtype=dtype, chunks=chunks)

        # Create a DataArray for the variable. We will set coords in the second pass
        dim_names = _get_dimension_names(v)
        data_array = xr.DataArray(arr, dims=dim_names) 

        # https://docs.xarray.dev/en/stable/internals/zarr-encoding-spec.html#zarr-encoding
        # If you don't explicitly specify a compressor when creating a Zarr array, Z
        # arr will use a default compressor based on the Zarr format version and the data type of your array. 
        # Zarr V2 (Default is Blosc)
        # data_array.encoding.compressor = None
        #TODO: beging the par that does not work
        data_array.encoding["fill_value"] = 0.0
        data_array.encoding["dimension_separator"] = "/" # Does not work
        if v.compressor is not None:
            compressor = _to_dictionary(v.compressor)
            data_array.encoding["compressor"] = compressor
        else:
            data_array.encoding["compressor"] = None
        #TODO: end the part that does not work

        # Add array attributes
        if v.metadata is not None:
            meta_dict = _to_dictionary(v.metadata)
            # Exclude chunk_grid
            del meta_dict["chunkGrid"]  
            # Remove empty attributes
            meta_dict = {k: v for k, v in meta_dict.items() if v is not None}
            # Add metadata to the data array attributes
            data_array.attrs.update(meta_dict)
            pass
        if v.long_name:
            data_array.attrs["long_name"] = v.long_name

        # Let's store the data array for the second pass
        data_arrays[v.name] = data_array

    # Add non-dimension coordinates to the data arrays
    for v in ds.variables:
        da = data_arrays[v.name]
        non_dim_coords_names = set(_get_coord_names(v)) - set(_get_dimension_names(v)) - {v.name}
        # Create a populate a dictionary {coord_name: DataArray for the coordinate}
        non_dim_coords_dict : dict[str, xr.DataArray] = {} 
        for name in non_dim_coords_names:
            non_dim_coords_dict[name] = data_arrays[name]
        if non_dim_coords_dict:
            # NOTE: here is a gotcha: assign_coords() does not update in-place, 
            # but returns an updated instance!
            data_arrays[v.name] = da.assign_coords(non_dim_coords_dict)
            pass

    # Now let's create a dataset with all data arrays
    xr_ds = xr.Dataset(data_arrays)
    # Attach dataset metadata
    if ds.metadata is not None:
        xr_ds.attrs["apiVersion"] = ds.metadata.api_version
        xr_ds.attrs["createdOn"] = str(ds.metadata.created_on)
        xr_ds.attrs["name"] = ds.metadata.name
        if ds.metadata.attributes:
            xr_ds.attrs["attributes"] = ds.metadata.attributes

    return xr_ds


def to_zarr(dataset: xr.Dataset,
    store: str | None = None,
    *args: str | int | float | bool,
    **kwargs: Mapping[str, str | int | float | bool],
) -> None:
    """Alias for `.to_zarr()`."""
    # Ensure zarr_format=2 by default unless explicitly overridden
    zarr_format = kwargs.get("zarr_format", 2)
    if zarr_format != 2:  # noqa: PLR2004
        msg = "MDIO only supports zarr_format=2"
        raise ValueError(msg)

    # ds.to_zarr("foo.zarr", consolidated=False, encoding={"foo": {"compressors": [compressor]}})
    # Define compressor
    # compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=2)
    # # Define encoding
    # encoding = {
    #     "foo": {"compressor": compressor},
    #     "bar": {"compressor": compressor},
    # }
    # kwargs["encoding"] = encoding
    encoding = {}
    for key in dataset.data_vars.keys():
        c = dataset[key].encoding.get("compressors", None)
        encoding[key] = {"compressors": c}
    kwargs["encoding"] = encoding

    kwargs["zarr_format"] = zarr_format




    return dataset.to_zarr(*args, store=store, **kwargs)

# https://docs.xarray.dev/en/stable/user-guide/io.html
# ds.to_zarr("path/to/directory.zarr", zarr_format=2, consolidated=False)