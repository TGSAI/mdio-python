"""MDIO accessor APIs."""

from __future__ import annotations

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
from mdio.core.exceptions import MDIONotFoundError
from mdio.exceptions import ShapeError


class MDIOAccessor:
    """Accessor class for MDIO files.

    The accessor can be used to read and write MDIO files. It allows you to
    open an MDIO file in several `mode` and `access_pattern` combinations.

    Access pattern defines the dimensions that are chunked. For instance
    if you have a 3-D array that is chunked in every direction (i.e. a
    3-D seismic stack consisting of inline, crossline, and sample dimensions)
    its access pattern would be "012". If it was only chunked in the first
    two dimensions (i.e. seismic inline and crossline), it would be "01".

    By default, MDIO will try to open with "012" access pattern, and will
    raise an error if that pattern doesn't exist.

    After dataset is opened, when the accessor is sliced it will either return
    just seismic trace data as a Numpy array or a tuple of live mask, headers,
    and seismic trace in Numpy based on the parameter `return_metadata`.

    Regarding object store access, if the user credentials have been set
    system-wide on local machine or VM; there is no need to specify credentials.
    However, the `storage_options` option allows users to specify credentials
    for the store that is being accessed. Please see the `fsspec` documentation
    for configuring storage options.

    MDIO currently supports `Zarr` and `Dask` backends. The Zarr backend
    is useful for reading small amounts of data with minimal overhead. However,
    by utilizing the `Dask` backend with a larger chunk size using the
    `new_chunks` argument, the data can be read in parallel using a Dask
    LocalCluster or a distributed Cluster.

    The accessor also allows users to enable `fsspec` caching. These are
    particularly useful when we are accessing the data from a high-latency
    store such as object stores, or mounted network drives with high latency.
    We can use the `disk_cache` option to fetch chunks the local temporary
    directory for faster repetitive access. We can also turn on the Least
    Recently Used (LRU) cache  by using the `memory_cache` option. It has
    to be specified in bytes.

    Args:
        mdio_path_or_buffer: Store URL for MDIO file. This can be either on
            a local disk, or a cloud object store.
        mode: Read or read/write mode. The file must exist. Options are
            in {'r', 'r+', 'w'}. 'r' is read only, 'r+' is append mode where
            only existing arrays can be modified, 'w' is similar to 'r+'
            but rechunking or other file-wide operations are allowed.
        access_pattern: Chunk access pattern, optional. Default is "012".
            Examples: '012', '01', '01234'.
        storage_options: Options for the storage backend. By default,
            system-wide credentials will be used. If system-wide credentials
            are not set and the source is not public, an authentication
            error will be raised by the backend.
        return_metadata: Flag for returning live mask, headers, and traces
            or just the trace data. Default is False, which means just trace
            data will be returned.
        new_chunks: Chunk sizes used in Dask backend. Ignored for Zarr
            backend. By default, the disk-chunks will be used. However, if
            we want to stream groups of chunks to a Dask worker, we can
            rechunk here. Then each Dask worker can asynchronously fetch
            multiple chunks before working.
        backend: Backend selection, optional. Default is "zarr". Must be
            in {'zarr', 'dask'}.
        memory_cache_size: Maximum, in memory, least recently used (LRU)
            cache size in bytes.
        disk_cache: Disk cache implemented by `fsspec`, optional. Default is
            False, which turns off disk caching. See `simplecache` from
            `fsspec` documentation for more details.

    Raises:
        MDIONotFoundError: If the MDIO file can not be opened.

    Notes:
        The combination of the `Dask` backend and caching schemes are experimental.
        This configuration may cause unexpected memory usage and duplicate data
        fetching.

    Examples:
        Assuming we ingested `my_3d_seismic.segy` as `my_3d_seismic.mdio` we can
        open the file in read-only mode like this.

        >>> from mdio import MDIOReader
        >>>
        >>>
        >>> mdio = MDIOReader("my_3d_seismic.mdio")

        This will open the file with the lazy `Zarr` backend. To access a
        specific inline, crossline, or sample index we can do:

        >>> inline = mdio[15]  # get the 15th inline
        >>> crossline = mdio[:, 15]  # get the 50th crossline
        >>> samples = mdio[..., 250]  # get the 250th sample slice

        The above will variables will be Numpy arrays of the relevant
        trace data. If we want to retreive the live mask and trace headers
        for our sliding we need to open the file with the `return_metadata`
        option.

        >>> mdio = MDIOReader("my_3d_seismic.mdio", return_metadata=True)

        Then we can fetch the data like this (for inline):

        >>> il_live, il_headers, il_traces = mdio[15]

        Since MDIOAccessor returns a tuple with these three Numpy arrays,
        we can directly unpack it and use it further down our code.
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
        storage_options: dict | None,
        return_metadata: bool,
        new_chunks: tuple[int, ...] | None,
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
            if self.mode in {"r", "r+"}:
                self.root = zarr.open_consolidated(store=self.store, mode=self.mode)
            elif self.mode == "w":
                self.root = zarr.open(store=self.store, mode="r+")
            else:
                msg = f"Invalid mode: {self.mode}"
                raise ValueError(msg)
        except KeyError as e:
            msg = (
                f"MDIO file not found or corrupt at {self.store.path}. "
                "Please check the URL or ensure it is not a deprecated "
                "version of MDIO file."
            )
            raise MDIONotFoundError(msg) from e

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
            header_kwargs["chunks"] = self.chunks[:-1]

        self._headers = self._array_loader(**header_kwargs)

        self.grid.live_mask = self._array_loader(self._metadata_group, name="live_mask")
        self.live_mask = self.grid.live_mask

    @property
    def live_mask(self) -> npt.ArrayLike | da.Array:
        """Get live mask (i.e. not-null value mask)."""
        return self._live_mask

    @live_mask.setter
    def live_mask(self, value: npt.ArrayLike | da.Array) -> None:
        """Set live mask (i.e. not-null value mask)."""
        self._live_mask = value

    @property
    def n_dim(self) -> int:
        """Get number of dimensions for dataset."""
        return self._n_dim

    @n_dim.setter
    def n_dim(self, value: int) -> None:
        """Set number of dimensions for dataset."""
        self._n_dim = value

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of dataset."""
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int, ...]) -> None:
        """Validate and set shape of dataset."""
        if not isinstance(value, tuple):
            raise AttributeError("Array shape needs to be a tuple")
        self._shape = value

    @property
    def trace_count(self) -> int:
        """Get trace count from seismic MDIO."""
        return self._trace_count

    @trace_count.setter
    def trace_count(self, value: int) -> None:
        """Validate and set trace count for seismic MDIO."""
        if not isinstance(value, int):
            raise AttributeError("Live trace count needs to be an integer")
        self._trace_count = value

    @property
    def text_header(self) -> list:
        """Get seismic text header."""
        return self._text_header

    @text_header.setter
    def text_header(self, value: list) -> None:
        """Validate and set seismic text header."""
        if not isinstance(value, list):
            raise AttributeError("Text header must be a list of str with 40 elements")
        self._text_header = value

    @property
    def binary_header(self) -> dict:
        """Get seismic binary header metadata."""
        return self._binary_header

    @binary_header.setter
    def binary_header(self, value: dict) -> None:
        """Validate and set seismic binary header metadata."""
        if not isinstance(value, dict):
            raise AttributeError("Binary header has to be a dictionary type collection")
        self._binary_header = value

    @property
    def chunks(self) -> tuple[int, ...]:
        """Get dataset chunk sizes."""
        return self._chunks

    @chunks.setter
    def chunks(self, value: tuple[int, ...]) -> None:
        """Set dataset chunk sizes."""
        self._chunks = value

    @property
    def stats(self) -> dict:
        """Get global statistics like min/max/rms/std."""
        return self._stats

    @stats.setter
    def stats(self, value: dict) -> None:
        """Set global statistics like min/max/rms/std."""
        self._stats = value

    @property
    def _metadata_group(self) -> zarr.Group:
        """Get metadata zarr.group handle."""
        return self.root.metadata

    @property
    def _data_group(self) -> zarr.Group:
        """Get data zarr.Group handle."""
        return self.root.data

    def __getitem__(self, item: int | tuple) -> npt.ArrayLike | da.Array | tuple:
        """Data getter."""
        if self._return_metadata is True:
            if isinstance(item, (int, slice)):
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
    ) -> tuple[NDArray[int], ...]:
        """Convert dimension coordinate to zero-based index.

        The coordinate labels of the array dimensions are converted to
        zero-based indices. For instance if we have an inline dimension like
        this:

        `[10, 20, 30, 40, 50]`

        then the indices would be:

        `[0, 1, 2, 3, 4]`

        This method converts from coordinate labels of a dimension to
        equivalent indices.

        Multiple dimensions can be queried at the same time, see the examples.

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
            Opening an MDIO file.

            >>> from mdio import MDIOReader
            >>>
            >>>
            >>> mdio = MDIOReader("path_to.mdio")
            >>> mdio.coord_to_index([10, 7, 15], dimensions='inline')
            array([ 8,  5, 13], dtype=uint16)

            >>> ils, xls = [10, 7, 15], [5, 10]
            >>> mdio.coord_to_index(ils, xls, dimensions=['inline', 'crossline'])
            (array([ 8,  5, 13], dtype=uint16), array([3, 8], dtype=uint16))

            With the above indices, we can slice the data:

            >>> mdio[ils]  # only inlines
            >>> mdio[:, xls]  # only crosslines
            >>> mdio[ils, xls]  # intersection of the lines

            Note that some fancy-indexing may not work with Zarr backend.
            The Dask backend is more flexible when it comes to indexing.

            If we are querying all dimensions of a 3D array, we can omit the
            `dimensions` argument.

            >>> mdio.coord_to_index(10, 5, [50, 100])
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
        for mdio_dim, dim_query_coords in zip(dims, queries):  # noqa: B905
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
            dim_idx = dim_idx.astype("uint32")  # cast max: 2,147,483,647
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

        Args:
            dest_path_or_buffer: Destination path. Could be any FSSpec mapping.
            excludes: Data to exclude during copy. i.e. `chunked_012`. The raw data
                won't be copied, but it will create an empty array to be filled.
                If left blank, it will copy everything.
            includes: Data to include during copy. i.e. `trace_headers`. If this is
                not specified, and certain data is excluded, it will not copy headers.
                If you want to preserve headers, specify `trace_headers`. If left blank,
                it will copy everything except specified in `excludes` parameter.
            storage_options: Storage options for the cloud storage backend.
                Default is None (will assume anonymous).
            overwrite: Overwrite destination or not.
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

    For detailed documentation see MDIOAccessor.

    Args:
        mdio_path_or_buffer: Store URL for MDIO file. This can be either on
            a local disk, or a cloud object store.
        access_pattern: Chunk access pattern, optional. Default is "012".
            Examples: '012', '01', '01234'.
        storage_options: Options for the storage backend. By default,
            system-wide credentials will be used. If system-wide credentials
            are not set and the source is not public, an authentication
            error will be raised by the backend.
        return_metadata: Flag for returning live mask, headers, and traces
            or just the trace data. Default is False, which means just trace
            data will be returned.
        new_chunks: Chunk sizes used in Dask backend. Ignored for Zarr
            backend. By default, the disk-chunks will be used. However, if
            we want to stream groups of chunks to a Dask worker, we can
            rechunk here. Then each Dask worker can asynchronously fetch
            multiple chunks before working.
        backend: Backend selection, optional. Default is "zarr". Must be
            in {'zarr', 'dask'}.
        memory_cache_size: Maximum, in memory, least recently used (LRU)
            cache size in bytes.
        disk_cache: Disk cache implemented by `fsspec`, optional. Default is
            False, which turns off disk caching. See `simplecache` from
            `fsspec` documentation for more details.
    """

    def __init__(
        self,
        mdio_path_or_buffer: str,
        access_pattern: str = "012",
        storage_options: dict = None,
        return_metadata: bool = False,
        new_chunks: tuple[int, ...] = None,
        backend: str = "zarr",
        memory_cache_size: int = 0,
        disk_cache: bool = False,
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
    """Writable accessor for MDIO files.

    For detailed documentation see MDIOAccessor.

    Args:
        mdio_path_or_buffer: Store URL for MDIO file. This can be either on
            a local disk, or a cloud object store.
        access_pattern: Chunk access pattern, optional. Default is "012".
            Examples: '012', '01', '01234'.
        storage_options: Options for the storage backend. By default,
            system-wide credentials will be used. If system-wide credentials
            are not set and the source is not public, an authentication
            error will be raised by the backend.
        return_metadata: Flag for returning live mask, headers, and traces
            or just the trace data. Default is False, which means just trace
            data will be returned.
        new_chunks: Chunk sizes used in Dask backend. Ignored for Zarr
            backend. By default, the disk-chunks will be used. However, if
            we want to stream groups of chunks to a Dask worker, we can
            rechunk here. Then each Dask worker can asynchronously fetch
            multiple chunks before working.
        backend: Backend selection, optional. Default is "zarr". Must be
            in {'zarr', 'dask'}.
        memory_cache_size: Maximum, in memory, least recently used (LRU)
            cache size in bytes.
        disk_cache: Disk cache implemented by `fsspec`, optional. Default is
            False, which turns off disk caching. See `simplecache` from
            `fsspec` documentation for more details.
    """

    def __init__(
        self,
        mdio_path_or_buffer: str,
        access_pattern: str = "012",
        storage_options: dict = None,
        return_metadata: bool = False,
        new_chunks: tuple[int, ...] = None,
        backend: str = "zarr",
        memory_cache_size: int = 0,
        disk_cache: bool = False,
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
