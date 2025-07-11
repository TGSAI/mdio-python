"""Convert MDIO v1 schema Dataset to Xarray DataSet and write it in Zarr."""

from collections.abc import Mapping

from dask import array as dask_array
from numcodecs import Blosc as nc_Blosc
from numpy import dtype as np_dtype
from numpy import nan as np_nan
from xarray import DataArray as xr_DataArray
from xarray import Dataset as xr_Dataset
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding

try:
    # zfpy is an optional dependency for ZFP compression
    # It is not installed by default, so we check for its presence and import it only if available.
    from zfpy import ZFPY as zfpy_ZFPY  # noqa: N811
except ImportError:
    zfpy_ZFPY = None  # noqa: N816

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
    all_named_dims: dict[str, NamedDimension] = {}
    for v in dataset.variables:
        if v.dimensions is not None:
            for d in v.dimensions:
                if isinstance(d, NamedDimension):
                    all_named_dims[d.name] = d
                else:
                    # Never happens for the dataset generated with the dataset builder
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
                # Never happens for the dataset generated with the dataset builder
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


def _get_np_datatype(var: Variable) -> np_dtype:
    data_type = var.data_type
    if isinstance(data_type, ScalarType):
        return np_dtype(data_type.value)
    if isinstance(data_type, StructuredType):
        return np_dtype([(f.name, f.format.value) for f in data_type.fields])
    err = f"Unsupported data type: {type(data_type)} in variable {var.name}"
    raise TypeError(err)


def _get_zarr_shape(var: Variable) -> tuple[int, ...]:
    # NOTE: This assumes that the variable dimensions are all NamedDimension
    return tuple(dim.size for dim in var.dimensions)


def _get_zarr_chunks(var: Variable) -> tuple[int, ...]:
    """Get the chunk shape for a variable, defaulting to its shape if no chunk grid is defined."""
    if var.metadata is not None and var.metadata.chunk_grid is not None:
        return var.metadata.chunk_grid.configuration.chunk_shape
    # Default to full shape if no chunk grid is defined
    return _get_zarr_shape(var)


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


# Do we already have it somewhere in the codebase? I could not find it.
fill_value_map = {
    ScalarType.BOOL: None,
    ScalarType.FLOAT16: np_nan,
    ScalarType.FLOAT32: np_nan,
    ScalarType.FLOAT64: np_nan,
    ScalarType.UINT8: 2**8 - 1,  # Max value for uint8
    ScalarType.UINT16: 2**16 - 1,  # Max value for uint16
    ScalarType.UINT32: 2**32 - 1,  # Max value for uint32
    ScalarType.UINT64: 2**64 - 1,  # Max value for uint64
    ScalarType.INT8: 2**7 - 1,  # Max value for int8
    ScalarType.INT16: 2**15 - 1,  # Max value for int16
    ScalarType.INT32: 2**31 - 1,  # Max value for int32
    ScalarType.INT64: 2**63 - 1,  # Max value for int64
    ScalarType.COMPLEX64: complex(np_nan, np_nan),
    ScalarType.COMPLEX128: complex(np_nan, np_nan),
    ScalarType.COMPLEX256: complex(np_nan, np_nan),
}


def _get_fill_value(data_type: ScalarType | StructuredType | str) -> any:
    """Get the fill value for a given data type.

    The Zarr fill_value is a scalar value providing the default value to use for
    uninitialized portions of the array, or null if no fill_value is to be used
    https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """
    if isinstance(data_type, ScalarType):
        return fill_value_map.get(data_type)
    if isinstance(data_type, StructuredType):
        return "AAAAAAAAAAAAAAAA"  # BUG: this does not work!!!
    if isinstance(data_type, str):
        return ""
    # If we do not have a fill value for this type, use None
    return None


def to_xarray_dataset(mdio_ds: Dataset) -> xr_DataArray:  # noqa: PLR0912
    """Build an XArray dataset with correct dimensions and dtypes.

    This function constructs the underlying data structure for an XArray dataset,
    handling dimension mapping, data types, and metadata organization.

    Args:
        mdio_ds: The source MDIO dataset to construct from.

    Returns:
        The constructed dataset with proper MDIO structure and metadata.
    """
    # See the xarray tutorial for more details on how to create datasets:
    # https://tutorial.xarray.dev/fundamentals/01.1_creating_data_structures.html

    # all_dims = _get_all_named_dimensions(ds)
    # all_coords = _get_all_coordinates(ds)

    # First pass: Build all variables
    data_arrays: dict[str, xr_DataArray] = {}
    for v in mdio_ds.variables:
        # Use dask array instead of numpy array for lazy evaluation
        shape = _get_zarr_shape(v)
        dtype = _get_np_datatype(v)
        chunks = _get_zarr_chunks(v)
        arr = dask_array.zeros(shape, dtype=dtype, chunks=chunks)

        # Create a DataArray for the variable. We will set coords in the second pass
        dim_names = _get_dimension_names(v)
        data_array = xr_DataArray(arr, dims=dim_names)

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

        # Compression:
        # https://docs.xarray.dev/en/stable/internals/zarr-encoding-spec.html#zarr-encoding
        # If you don't explicitly specify a compressor when creating a Zarr array, Z
        # arr will use a default compressor based on the Zarr format version and the data
        # type of your array. Zarr V2 (Default is Blosc).
        # Thus, if there is no compressor, we will explicitly set "compressor" to None.
        #
        # Create a custom chunk key encoding with "/" as separator
        chunk_key_encoding = V2ChunkKeyEncoding(separator="/").to_dict()
        encoding = {
            "fill_value": _get_fill_value(v.data_type),
            "chunks": chunks,
            "chunk_key_encoding": chunk_key_encoding,
            # I was hoping the following would work, but it does not.
            # I see:
            #    >_compressor = parse_compressor(compressor[0])
            #    > return numcodecs.get_codec(data)
            #    E - numcodecs.errors.UnknownCodecError: codec not available: 'None'"
            # Example: https://zarr.readthedocs.io/en/stable/user-guide/arrays.html#compressors
            # from zarr.codecs import BloscCodec
            # compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
            #
            # "compressor": _to_dictionary(v.compressor)
            # Thus, we will call the conversion function:
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
) -> None:
    """Write an XArray dataset to Zarr format."""
    # MDIO only supports zarr_format=2
    kwargs["zarr_format"] = 2
    # compute: default: True) – If True write array data immediately,
    # otherwise return a dask.delayed.Delayed object that can be computed
    # to write array data later.
    # *** Metadata is always updated eagerly. ***
    kwargs["compute"] = False
    # https://docs.xarray.dev/en/stable/user-guide/io.html
    # https://docs.xarray.dev/en/latest/generated/xarray.DataArray.to_zarr.html
    return dataset.to_zarr(*args, store=store, **kwargs)
