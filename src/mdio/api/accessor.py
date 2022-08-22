"""MDIO accessor APIs."""


from __future__ import annotations

from warnings import simplefilter
from warnings import warn

import dask.array as da
import numpy as np
import numpy.typing as npt
import zarr
from numpy.typing import NDArray

from mdio.api.convenience import copy_mdio
from mdio.api.io_utils import open_zarr_array
from mdio.api.io_utils import open_zarr_array_dask
from mdio.api.io_utils import process_url
from mdio.core import Grid
from mdio.exceptions import ShapeError


simplefilter("always", DeprecationWarning)


class MDIOAccessor:
    """Accessor class for MDIO files.

    Args:
        mdio_path_or_buffer: Store URL for MDIO file
        mode: Read or read/write (must exist), in {'r', 'r+'}
        access_pattern: Chunk access pattern. Examples: '012', '01'
        storage_options: Options for the storage backend. Default is `None` (anonymous)
        return_metadata: Flag for returning metadata + traces or just traces
        new_chunks: Custom chunksizes used in 'dask' backend. Ignored for 'zarr` backend
        backend: Eager `zarr` or lazy but more flexible `dask` backend
        memory_cache_size: Maximum in memory LRU cache size in bytes
        disk_cache: FSSpec's `simplecache` if True. Default is False.
    """

    _array_load_function_mapper = {
        "zarr": open_zarr_array,
        "dask": open_zarr_array_dask,
    }

    def __init__(
        self,
        mdio_path_or_buffer: str,
        mode: str,
        access_pattern: str,
        storage_options: dict,
        return_metadata: bool,
        new_chunks: tuple[int, ...],
        backend: str,
        memory_cache_size: int,
        disk_cache: bool,
    ):
        """Accessor initialization function."""
        # Set public attributes
        self.url = mdio_path_or_buffer
        self.mode = mode
        self.access_pattern = access_pattern

        # Set private attributes for public interface.
        # Pep8 complains because they are defined outside __init__
        self._binary_header = None
        self._chunks = None
        self._live_mask = None
        self._root = None
        self._n_dim = None
        self._orig_chunks = None
        self._store = None
        self._shape = None
        self._stats = None
        self._text_header = None
        self._trace_count = None

        # Private attributes
        self._array_loader = self._array_load_function_mapper[backend]
        self._backend = backend
        self._return_metadata = return_metadata
        self._new_chunks = new_chunks
        self._memory_cache_size = memory_cache_size
        self._disk_cache = disk_cache

        # Call methods to finish initialization
        self._validate_store(storage_options)
        self._connect()
        self._deserialize_grid()
        self._set_attributes()
        self._open_arrays()

    def _validate_store(self, storage_options):
        """Method to validate the provided store."""
        if storage_options is None:
            storage_options = {}

        self.store = process_url(
            url=self.url,
            mode=self.mode,
            storage_options=storage_options,
            memory_cache_size=self._memory_cache_size,
            disk_cache=self._disk_cache,
        )

    def _connect(self):
        """Open the zarr root."""
        try:
            self.root = zarr.open_consolidated(
                store=self.store,
                mode=self.mode,
                metadata_key="zmetadata",
            )
        except KeyError:
            # Backwards compatibility pre v0.1.0
            # This will be irrelevant when we go zarr v3.
            self.store.key_separator = "."
            self.root = zarr.open_consolidated(
                store=self.store,
                mode=self.mode,
            )
            msg = (
                "Encountered an older MDIO file (pre MDIO). The "
                "support for these files will be removed at a future release. "
                "Please consider re-ingesting your files with the latest "
                "version of MDIO to avoid problems in the future."
            )
            warn(msg, DeprecationWarning, stacklevel=2)

    def _deserialize_grid(self):
        """Deserialize grid from Zarr metadata."""
        self.grid = Grid.from_zarr(self.root)

    def _set_attributes(self):
        """Deserialize attributes from Zarr metadata."""
        self.trace_count = self.root.attrs["trace_count"]
        self.stats = {
            key: self.root.attrs[key] for key in ["mean", "std", "rms", "min", "max"]
        }

        self.text_header = self._metadata_group.attrs["text_header"]
        self.binary_header = self._metadata_group.attrs["binary_header"]

        # Grid based attributes
        self.shape = self.grid.shape
        self.n_dim = len(self.shape)

        # Access pattern attributes
        data_array_name = "_".join(["chunked", self.access_pattern])
        self.chunks = self._data_group[data_array_name].chunks
        self._orig_chunks = self.chunks

        if self._backend == "dask" and self._new_chunks is not None:
            # Handle None values (take original chunksize)
            new_chunks = tuple(
                self.chunks[idx] if dim is None else dim
                for idx, dim in enumerate(self._new_chunks)
            )

            # Handle "-1" values, which means don't chunk that dimension
            new_chunks = tuple(
                self.shape[idx] if dim == -1 else dim
                for idx, dim in enumerate(new_chunks)
            )

            print(f"Array shape is {self.shape}")
            self._orig_chunks = self.chunks
            self.chunks = new_chunks

    def _open_arrays(self):
        """Open arrays with requested backend."""
        data_array_name = "_".join(["chunked", self.access_pattern])
        header_array_name = "_".join(["chunked", self.access_pattern, "trace_headers"])

        trace_kwargs = dict(
            group_handle=self._data_group,
            name=data_array_name,
        )

        if self._backend == "dask":
            trace_kwargs["chunks"] = self.chunks

        self._traces = self._array_loader(**trace_kwargs)

        if self._backend == "dask" and self._orig_chunks != self._chunks:
            dask_chunksize = self._traces.chunksize
            print(f"Setting (dask) chunks from {self._orig_chunks} to {dask_chunksize}")
            self.chunks = dask_chunksize

        header_kwargs = dict(
            group_handle=self._metadata_group,
            name=header_array_name,
        )

        if self._backend == "dask":
            trace_kwargs["chunks"] = self.chunks[:-1]

        self._headers = self._array_loader(**header_kwargs)

        self.grid.live_mask = self._array_loader(self._metadata_group, name="live_mask")
        self.live_mask = self.grid.live_mask

    @property
    def live_mask(self) -> npt.ArrayLike | da.Array:
        """Get live mask."""
        return self._live_mask

    @live_mask.setter
    def live_mask(self, value: npt.ArrayLike | da.Array) -> None:
        """Set live mask."""
        self._live_mask = value

    @property
    def n_dim(self) -> int:
        """Get dimensionality."""
        return self._n_dim

    @n_dim.setter
    def n_dim(self, value: int) -> None:
        """Set dimensionality."""
        self._n_dim = value

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape."""
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int, ...]) -> None:
        """Validate and set shape."""
        if not isinstance(value, tuple):
            raise AttributeError("Array shape needs to be a tuple")
        self._shape = value

    @property
    def trace_count(self) -> int:
        """Get trace count."""
        return self._trace_count

    @trace_count.setter
    def trace_count(self, value: int) -> None:
        """Validate and set trace count."""
        if not isinstance(value, int):
            raise AttributeError("Live trace count needs to be an integer")
        self._trace_count = value

    @property
    def text_header(self) -> list:
        """Get text header."""
        return self._text_header

    @text_header.setter
    def text_header(self, value: list) -> None:
        """Validate and set text header."""
        if not isinstance(value, list):
            raise AttributeError("Text header must be a list of str with 40 elements")
        self._text_header = value

    @property
    def binary_header(self) -> dict:
        """Get binary header."""
        return self._binary_header

    @binary_header.setter
    def binary_header(self, value: dict) -> None:
        """Validate and set binary header."""
        if not isinstance(value, dict):
            raise AttributeError("Binary header has to be a dictionary type collection")
        self._binary_header = value

    @property
    def chunks(self) -> tuple[int, ...]:
        """Get chunk sizes."""
        return self._chunks

    @chunks.setter
    def chunks(self, value: tuple[int, ...]) -> None:
        """Set chunk sizes."""
        self._chunks = value

    @property
    def stats(self) -> dict:
        """Get global statistics."""
        return self._stats

    @stats.setter
    def stats(self, value: dict) -> None:
        """Set global statistics."""
        self._stats = value

    @property
    def _metadata_group(self) -> zarr.Group:
        """Get metadata group handle."""
        return self.root.metadata

    @property
    def _data_group(self) -> zarr.Group:
        """Get data group handle."""
        return self.root.data

    def __getitem__(self, item: int | tuple) -> npt.ArrayLike | da.Array | tuple:
        """Data gettter."""
        if self._return_metadata is True:
            if type(item) == int or type(item) == slice:
                meta_index = item
            elif len(item) == len(self.shape):
                meta_index = tuple(dim for dim in item[:-1])
            else:
                meta_index = item

            return (
                self.live_mask[meta_index],
                self._headers[meta_index],
                self._traces[item],
            )

        return self._traces[item]

    def __setitem__(self, key: int | tuple, value: npt.ArrayLike) -> None:
        """Data setter."""
        self._traces[key] = value

    def coord_to_index(
        self,
        *args,
        dimensions: str | list[str] | None = None,
    ) -> tuple[NDArray[np.int], ...]:
        """Convert dimension coordinate to zero-based index.

        Args:
            *args: Variable length argument queries.  # noqa: RST213
            dimensions: Name of the dimensions to query. If not provided, it
                will query all dimensions in the grid and will require
                `len(args) == grid.ndim`

        Returns:
            Zero-based indices of coordinates. Each item in result corresponds
            to indicies of that dimension

        Raises:
            ShapeError: if number of queries don't match requested dimensions.
            ValueError: if requested coordinates don't exist.

        Examples:
            >>> reader = MDIOReader("path_to.mdio")
            >>> reader.coord_to_index([10, 7, 15], dimensions='inline')
            array([ 8,  5, 13], dtype=uint16)

            >>> ils, xls = [10, 7, 15], [5, 10]
            >>> reader.coord_to_index(ils, xls, dimensions=['inline', 'crossline'])
            (array([ 8,  5, 13], dtype=uint16), array([3, 8], dtype=uint16))

            Given 3-D Array

            >>> reader.coord_to_index(10, 5, [50, 100])
            (array([8], dtype=uint16),
             array([3], dtype=uint16),
             array([25, 50], dtype=uint16))

        """
        queries = [np.atleast_1d(dim_query) for dim_query in args]

        # Ensure dimensions is a list
        if dimensions is not None and not isinstance(dimensions, list):
            dimensions = [dimensions]

        # Ensure the query arrays and query dimensions match in size
        ndim_expect = self.grid.ndim if dimensions is None else len(dimensions)

        if len(queries) != ndim_expect:
            raise ShapeError(
                "Coordinate queries not the same size as n_dimensions",
                ("# Coord Dims", "# Dimensions"),
                (len(queries), ndim_expect),
            )

        if dimensions is None:
            dims = self.grid.dims
        else:
            dims = [self.grid.select_dim(dim_name) for dim_name in dimensions]

        dim_indices = tuple()
        for mdio_dim, dim_query_coords in zip(dims, queries):
            # Make sure all coordinates exist.
            query_diff = np.setdiff1d(dim_query_coords, mdio_dim.coords)
            if len(query_diff) > 0:
                msg = (
                    f"{mdio_dim.name} dimension does not have "
                    f"coordinate(s) {query_diff}"
                )
                raise ValueError(msg)

            sorter = mdio_dim.coords.argsort()
            dim_idx = np.searchsorted(mdio_dim, dim_query_coords, sorter=sorter)
            dim_idx = dim_idx.astype("uint16")  # cast to minimize memory. max: 65,535
            dim_indices += (dim_idx,)

        return dim_indices if len(dim_indices) > 1 else dim_indices[0]

    def copy(
        self,
        dest_path_or_buffer: str,
        excludes: str = "",
        includes: str = "",
        storage_options: dict | None = None,
        overwrite: bool = False,
    ):
        """Makes a copy of an MDIO file with or without all arrays.

        Refer to mdio.api.convenience.copy for full documentation.
        """
        copy_mdio(
            source=self,
            dest_path_or_buffer=dest_path_or_buffer,
            excludes=excludes,
            includes=includes,
            storage_options=storage_options,
            overwrite=overwrite,
        )


class MDIOReader(MDIOAccessor):
    """Read-only accessor for MDIO files.

    Args:
        mdio_path_or_buffer: Store URL for MDIO file
        access_pattern: Chunk access pattern. Examples: '012', '01'
        storage_options: Options for the storage backend. Default is `None` (anonymous)
        return_metadata: Flag for returning metadata + traces or just traces
        new_chunks: Custom chunksizes used in 'dask' backend. Ignored for 'zarr` backend
        backend: Eager `zarr` or lazy but more flexible `dask` backend
        memory_cache_size: Maximum in memory LRU cache size in bytes
        disk_cache: FSSpec's `simplecache` if True. Default is False.
    """

    def __init__(
        self,
        mdio_path_or_buffer: str,
        access_pattern: str = "012",
        storage_options: dict = None,
        return_metadata: bool = False,
        new_chunks: tuple[int, ...] = None,
        backend: str = "zarr",
        memory_cache_size=0,
        disk_cache=False,
    ):  # TODO: Disabled all caching by default, sometimes causes performance issues
        """Initialize super class with `r` permission."""
        super().__init__(
            mdio_path_or_buffer=mdio_path_or_buffer,
            mode="r",
            access_pattern=access_pattern,
            storage_options=storage_options,
            return_metadata=return_metadata,
            new_chunks=new_chunks,
            backend=backend,
            memory_cache_size=memory_cache_size,
            disk_cache=disk_cache,
        )


class MDIOWriter(MDIOAccessor):
    """Read-Write accessor for MDIO files.

    Args:
        mdio_path_or_buffer: Store URL for MDIO file
        access_pattern: Chunk access pattern. Examples: '012', '01'
        storage_options: Options for the storage backend. Default is `None` (anonymous)
        return_metadata: Flag for returning metadata + traces or just traces
        new_chunks: Custom chunksizes used in 'dask' backend. Ignored for 'zarr` backend
        backend: Eager `zarr` or lazy but more flexible `dask` backend
        memory_cache_size: Maximum in memory LRU cache size in bytes
        disk_cache: FSSpec's `simplecache` if True. Default is False.
    """

    def __init__(
        self,
        mdio_path_or_buffer: str,
        access_pattern: str = "012",
        storage_options: dict = None,
        return_metadata: bool = False,
        new_chunks: tuple[int, ...] = None,
        backend: str = "zarr",
        memory_cache_size=0,
        disk_cache=False,
    ):  # TODO: Disabled all caching by default, sometimes causes performance issues
        """Initialize super class with `r+` permission."""
        super().__init__(
            mdio_path_or_buffer=mdio_path_or_buffer,
            mode="r+",
            access_pattern=access_pattern,
            storage_options=storage_options,
            return_metadata=return_metadata,
            new_chunks=new_chunks,
            backend=backend,
            memory_cache_size=memory_cache_size,
            disk_cache=disk_cache,
        )
