"""Convert MDIO v1 schema Dataset to Xarray DataSet and write it in Zarr."""

from collections.abc import Mapping

import numpy as np
from numcodecs import Blosc as nc_Blosc
from numpy import dtype as np_dtype
from xarray import DataArray as xr_DataArray
from xarray import Dataset as xr_Dataset

import xarray
from zarr import zeros as zarr_zeros
import zarr
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from xarray.backends import ZarrStore
from dask.delayed import Delayed

from mdio.converters.type_converter import to_numpy_dtype

try:
    # zfpy is an optional dependency for ZFP compression
    # It is not installed by default, so we check for its presence and import it only if available.
    from zfpy import ZFPY as zfpy_ZFPY  # noqa: N811
except ImportError:
    zfpy_ZFPY = None  # noqa: N816

from mdio.constants import fill_value_map
from mdio.schemas.compressors import ZFP as mdio_ZFP  # noqa: N811
from mdio.schemas.compressors import Blosc as mdio_Blosc
from mdio.schemas.dimension import NamedDimension
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset_builder import _to_dictionary
from mdio.schemas.v1.variable import Coordinate
from mdio.schemas.v1.variable import Variable


def _get_all_named_dimensions(dataset: Dataset) -> dict[str, NamedDimension]:
    """Get all NamedDimensions from the dataset variables.

    This function returns a dictionary of NamedDimensions, but if some dimensions
    are not resolvable, they will not be included in the result.

    Args:
        dataset: The MDIO Dataset to extract NamedDimensions from.

    Note:
        The Dataset Builder ensures that all dimensions are resolvable by always embedding
        dimensions as NamedDimension and never as str.
        If the dataset is created in a different way, some dimensions may be specified as
        dimension names (str) instead of NamedDimension. In this case, we will try to resolve
        them to NamedDimension, but if the dimension is not found, it will be skipped.
        It is the responsibility of the Dataset creator to ensure that all dimensions are
        resolvable at the Dataset level.

    Returns:
        A dictionary mapping dimension names to NamedDimension instances.
    """
    all_named_dims: dict[str, NamedDimension] = {}
    for v in dataset.variables:
        if v.dimensions is not None:
            for d in v.dimensions:
                if isinstance(d, NamedDimension):
                    all_named_dims[d.name] = d
                else:
                    pass
    return all_named_dims


def _get_dimension_names(var: Variable) -> list[str]:
    """Get the names of dimensions for a variable.

    Note:
        We expect that Datasets produced by DatasetBuilder has all dimensions
        embedded as NamedDimension, but we also support dimension name strings for
        compatibility with Dataset produced in a different way.
    """
    dim_names: list[str] = []
    if var.dimensions is not None:
        for d in var.dimensions:
            if isinstance(d, NamedDimension):
                dim_names.append(d.name)
            elif isinstance(d, str):
                dim_names.append(d)
    return dim_names


def _get_coord_names(var: Variable) -> list[str]:
    """Get the names of coordinates for a variable."""
    coord_names: list[str] = []
    if var.coordinates is not None:
        for c in var.coordinates:
            if isinstance(c, Coordinate):
                coord_names.append(c.name)
            elif isinstance(c, str):
                coord_names.append(c)
    return coord_names

def _get_zarr_shape(var: Variable, all_named_dims: dict[str, NamedDimension]) -> tuple[int, ...]:
    """Get the shape of a variable for Zarr storage.

    Note:
        We expect that Datasets produced by DatasetBuilder has all dimensions
        embedded as NamedDimension, but we also support dimension name strings for
        compatibility with Dataset produced in a different way.
    """
    shape: list[int] = []
    for dim in var.dimensions:
        if isinstance(dim, NamedDimension):
            shape.append(dim.size)
        if isinstance(dim, str):
            named_dim = all_named_dims.get(dim)
            if named_dim is None:
                err = f"Dimension named '{dim}' can't be resolved to a NamedDimension."
                raise ValueError(err)
            shape.append(named_dim.size)
    return tuple(shape)


def _get_zarr_chunks(var: Variable, all_named_dims: dict[str, NamedDimension]) -> tuple[int, ...]:
    """Get the chunk shape for a variable, defaulting to its shape if no chunk grid is defined."""
    if var.metadata is not None and var.metadata.chunk_grid is not None:
        return tuple(var.metadata.chunk_grid.configuration.chunk_shape)
    # Default to full shape if no chunk grid is defined
    return _get_zarr_shape(var, all_named_dims=all_named_dims)


def _convert_compressor(
    compressor: mdio_Blosc | mdio_ZFP | None,
) -> nc_Blosc | zfpy_ZFPY | None:
    """Convert a compressor to a numcodecs compatible format."""
    if compressor is None:
        return None

    if isinstance(compressor, mdio_Blosc):
        return nc_Blosc(
            cname=compressor.algorithm.value,
            clevel=compressor.level,
            shuffle=compressor.shuffle.value,
            blocksize=compressor.blocksize if compressor.blocksize > 0 else 0,
        )

    if isinstance(compressor, mdio_ZFP):
        if zfpy_ZFPY is None:
            msg = "zfpy and numcodecs are required to use ZFP compression"
            raise ImportError(msg)
        return zfpy_ZFPY(
            mode=compressor.mode.value,
            tolerance=compressor.tolerance,
            rate=compressor.rate,
            precision=compressor.precision,
        )

    msg = f"Unsupported compressor model: {type(compressor)}"
    raise TypeError(msg)


def _get_fill_value(data_type: ScalarType | StructuredType | str) -> any:
    """Get the fill value for a given data type.

    The Zarr fill_value is a scalar value providing the default value to use for
    uninitialized portions of the array, or null if no fill_value is to be used
    https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """
    if isinstance(data_type, ScalarType):
        return fill_value_map.get(data_type)
    if isinstance(data_type, StructuredType):
        d_type = to_numpy_dtype(data_type)
        return np.zeros((), dtype=d_type)
    if isinstance(data_type, str):
        return ""
    # If we do not have a fill value for this type, use None
    return None


def to_xarray_dataset(mdio_ds: Dataset) -> xr_Dataset:  # noqa: PLR0912
    """Build an XArray dataset with correct dimensions and dtypes.

    This function constructs the underlying data structure for an XArray dataset,
    handling dimension mapping, data types, and metadata organization.

    Args:
        mdio_ds: The source MDIO dataset to construct from.

    Notes:
        - We can't use Dask (e.g., dask_array.zeros) because of the problems with
          structured type support. We will uze zarr.zeros instead

    Returns:
        The constructed dataset with proper MDIO structure and metadata.
    """
    # See the xarray tutorial for more details on how to create datasets:
    # https://tutorial.xarray.dev/fundamentals/01.1_creating_data_structures.html

    all_named_dims = _get_all_named_dimensions(mdio_ds)

    # First pass: Build all variables
    data_arrays: dict[str, xr_DataArray] = {}
    for v in mdio_ds.variables:
        shape = _get_zarr_shape(v, all_named_dims=all_named_dims)
        dtype = to_numpy_dtype(v.data_type)
        chunks = _get_zarr_chunks(v, all_named_dims=all_named_dims)

        # Use zarr.zeros to create an empty array with the specified shape and dtype
        # NOTE: zarr_format=2 is essential, to_zarr() will fail if zarr_format=2 is used
        data = zarr_zeros(shape=shape, dtype=dtype, zarr_format=2)
        # Create a DataArray for the variable. We will set coords in the second pass
        dim_names = _get_dimension_names(v)
        data_array = xr_DataArray(data, dims=dim_names)

        # Add array attributes
        if v.metadata is not None:
            meta_dict = _to_dictionary(v.metadata)
            # Exclude chunk_grid
            del meta_dict["chunkGrid"]
            # Remove empty attributes
            meta_dict = {k: v for k, v in meta_dict.items() if v is not None}
            # Add metadata to the data array attributes
            data_array.attrs.update(meta_dict)
        if v.long_name:
            data_array.attrs["long_name"] = v.long_name

        # Create a custom chunk key encoding with "/" as separator
        chunk_key_encoding = V2ChunkKeyEncoding(separator="/").to_dict()
        encoding = {
            # NOTE: See Zarr documentation on use of fill_value and _FillValue in Zarr v2 vs v3
            "_FillValue": _get_fill_value(v.data_type),
            "chunks": chunks,
            "chunk_key_encoding": chunk_key_encoding,
            "compressor": _convert_compressor(v.compressor),
        }
        data_array.encoding = encoding

        # Let's store the data array for the second pass
        data_arrays[v.name] = data_array

    # Second pass: Add non-dimension coordinates to the data arrays
    for v in mdio_ds.variables:
        da = data_arrays[v.name]
        non_dim_coords_names = set(_get_coord_names(v)) - set(_get_dimension_names(v)) - {v.name}
        # Create and populate a dictionary {coord_name: DataArray for the coordinate}
        non_dim_coords_dict: dict[str, xr_DataArray] = {}
        for name in non_dim_coords_names:
            non_dim_coords_dict[name] = data_arrays[name]
        if non_dim_coords_dict:
            # NOTE: here is a gotcha: assign_coords() does not update in-place,
            # but returns an updated instance!
            data_arrays[v.name] = da.assign_coords(non_dim_coords_dict)

    # Now let's create a dataset with all data arrays
    xr_ds = xr_Dataset(data_arrays)
    # Attach dataset metadata
    if mdio_ds.metadata is not None:
        xr_ds.attrs["apiVersion"] = mdio_ds.metadata.api_version
        xr_ds.attrs["createdOn"] = str(mdio_ds.metadata.created_on)
        xr_ds.attrs["name"] = mdio_ds.metadata.name
        if mdio_ds.metadata.attributes:
            xr_ds.attrs["attributes"] = mdio_ds.metadata.attributes
            

    return xr_ds


def to_zarr(
    dataset: xr_Dataset,
    store: str | None = None,
    *args: str | int | float | bool,
    **kwargs: Mapping[str, str | int | float | bool],
) -> ZarrStore | Delayed:
    """Write an XArray dataset to Zarr format.

    Args:
        dataset: The XArray dataset to write.
        store: The Zarr store to write to. If None, defaults to in-memory store.
        *args: Additional positional arguments for the Zarr store.
        **kwargs: Additional keyword arguments for the Zarr store.

    Notes:
        It sets the zarr_format to 2, which is the default for XArray datasets.
        Since we set kwargs["compute"], this method will return a dask.delayed.Delayed object
        and the arrays will not be immediately written.

    References:
            https://docs.xarray.dev/en/stable/user-guide/io.html
            https://docs.xarray.dev/en/latest/generated/xarray.DataArray.to_zarr.html

    Returns:
        None: The function writes the dataset as dask.delayed.Delayed object to the
        specified Zarr store.
    """
    kwargs["zarr_format"] = 2
    kwargs["compute"] = False
    kwargs["mode"] = "w"  # create (overwrite if exists)
    return dataset.to_zarr(*args, store=store, **kwargs)
