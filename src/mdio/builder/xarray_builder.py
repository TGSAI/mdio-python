"""Convert MDIO v1 schema Dataset to Xarray DataSet and write it in Zarr."""

import numcodecs
import numpy as np
import zarr
from dask import array as dask_array
from dask.array.core import normalize_chunks
from numcodecs import Blosc as numcodecs_Blosc
from xarray import DataArray as xr_DataArray
from xarray import Dataset as xr_Dataset
from zarr.codecs import BloscCodec as zarr_BloscCodec
from zarr.codecs.numcodecs import ZFPY as zarr_ZFPY  # noqa: N811

from mdio.builder.schemas.compressors import ZFP as mdio_ZFP  # noqa: N811
from mdio.builder.schemas.compressors import Blosc as mdio_Blosc
from mdio.builder.schemas.dimension import NamedDimension
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.v1.dataset import Dataset
from mdio.builder.schemas.v1.variable import Coordinate
from mdio.builder.schemas.v1.variable import Variable
from mdio.constants import ZarrFormat
from mdio.constants import fill_value_map
from mdio.converters.type_converter import to_numpy_dtype
from mdio.core.zarr_io import zarr_warnings_suppress_unstable_numcodecs_v3


def _import_numcodecs_zfpy() -> "type[numcodecs.ZFPY]":
    """Helper to import the optional dependency at runtime."""
    from numcodecs import ZFPY as numcodecs_ZFPY  # noqa: PLC0415, N811

    return numcodecs_ZFPY


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
        return var.metadata.chunk_grid.configuration.chunk_shape
    # Default to full shape if no chunk grid is defined
    return _get_zarr_shape(var, all_named_dims=all_named_dims)


def _compressor_to_encoding(
    compressor: mdio_Blosc | mdio_ZFP | None,
) -> dict[str, "zarr.codecs.Blosc | numcodecs.Blosc | numcodecs.ZFPY | zarr.codecs.ZFPY | None"] | None:
    """Convert a compressor to a numcodecs compatible format."""
    if compressor is None:
        return None

    if not isinstance(compressor, (mdio_Blosc, mdio_ZFP)):
        msg = f"Unsupported compressor model: {type(compressor)}"
        raise TypeError(msg)

    is_v2 = zarr.config.get("default_zarr_format") == ZarrFormat.V2
    kwargs = compressor.model_dump(exclude={"name"}, mode="json")

    if isinstance(compressor, mdio_Blosc):
        if is_v2 and kwargs["shuffle"] is None:
            kwargs["shuffle"] = -1
        codec_cls = numcodecs_Blosc if is_v2 else zarr_BloscCodec
        return {"compressors": codec_cls(**kwargs)}

    # must be ZFP beyond here
    try:
        numcodecs_ZFPY = _import_numcodecs_zfpy()  # noqa: N806
    except ImportError as e:
        msg = "The 'zfpy' package is required for lossy compression. Install via 'pip install multidimio[lossy]'."
        raise ImportError(msg) from e

    kwargs["mode"] = compressor.mode.int_code
    if is_v2:
        return {"compressors": numcodecs_ZFPY(**kwargs)}
    with zarr_warnings_suppress_unstable_numcodecs_v3():
        serializer = zarr_ZFPY(**kwargs)
    return {"serializer": serializer, "compressors": None}


def _get_fill_value(data_type: ScalarType | StructuredType | str) -> any:
    """Get the fill value for a given data type."""
    if isinstance(data_type, ScalarType):
        return fill_value_map.get(data_type)
    if isinstance(data_type, StructuredType):
        numpy_dtype = to_numpy_dtype(data_type)
        fill_value = (0,) * len(numpy_dtype.fields)
        return np.void(fill_value, dtype=numpy_dtype)
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
        - Using dask.array.zeros for lazy evaluation to prevent eager memory allocation
          while maintaining support for structured dtypes

    Returns:
        The constructed dataset with proper MDIO structure and metadata.
    """
    all_named_dims = _get_all_named_dimensions(mdio_ds)

    # First pass: Build all variables
    data_arrays: dict[str, xr_DataArray] = {}
    for v in mdio_ds.variables:
        # Retrieve the array shape, data type, and original chunk sizes
        shape = _get_zarr_shape(v, all_named_dims=all_named_dims)
        dtype = to_numpy_dtype(v.data_type)
        original_chunks = _get_zarr_chunks(v, all_named_dims=all_named_dims)

        # For efficient lazy array creation with Dask use larger chunks to minimize the task graph size
        # Initialize with original chunks for lazy array creation
        lazy_chunks = original_chunks
        if shape != original_chunks:
            # Compute automatic chunk sizes based on heuristics, respecting original chunks where possible
            auto_chunks = normalize_chunks("auto", shape=shape, dtype=dtype, previous_chunks=original_chunks)

            # Extract the primary (uniform) chunk size for each dimension, ignoring any variable remainder chunks
            uniform_auto = tuple(dim_chunks[0] for dim_chunks in auto_chunks)

            # Ensure creation chunks are at least as large as the original chunks to avoid splitting chunks
            lazy_chunks = tuple(max(auto, orig) for auto, orig in zip(uniform_auto, original_chunks, strict=True))

        data = dask_array.full(shape=shape, dtype=dtype, chunks=lazy_chunks, fill_value=_get_fill_value(v.data_type))

        # Create a DataArray for the variable. We will set coords in the second pass
        dim_names = _get_dimension_names(v)
        data_array = xr_DataArray(data, dims=dim_names)

        # Add array attributes
        if v.metadata is not None:
            metadata_dict = v.metadata.model_dump(exclude_none=True, mode="json", exclude={"chunk_grid"})
            data_array.attrs.update(metadata_dict)
        if v.long_name:
            data_array.attrs["long_name"] = v.long_name

        zarr_format = zarr.config.get("default_zarr_format")
        fill_value_key = "_FillValue" if zarr_format == ZarrFormat.V2 else "fill_value"
        fill_value = _get_fill_value(v.data_type) if v.name != "headers" else None

        encoding = {
            "chunks": original_chunks,
            fill_value_key: fill_value,
        }

        compressor_encodings = _compressor_to_encoding(v.compressor)

        if compressor_encodings is not None:
            encoding.update(compressor_encodings)

        if zarr_format == ZarrFormat.V2:
            encoding["chunk_key_encoding"] = {"name": "v2", "configuration": {"separator": "/"}}

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
