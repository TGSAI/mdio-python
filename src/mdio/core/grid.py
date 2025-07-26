"""Grid abstraction with serializers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mdio.core import Dimension
from mdio.core.serialization import Serializer

if TYPE_CHECKING:
    import zarr
    from segy.arrays import HeaderArray
    from zarr import Array as ZarrArray


@dataclass
class Grid:
    """N-dimensional grid class for managing bounds and increments.

    This class encapsulates an N-dimensional grid, storing dimension information and optional
    mapping and live mask arrays for trace indexing. It provides access to dimension names, shape,
    and number of dimensions as computed attributes.

    Args:
        dims: List of Dimension instances defining the grid axes.
        map: Optional Zarr array for trace mapping. Defaults to None.
        live_mask: Optional Zarr array indicating live traces. Defaults to None.

    Attributes:
        dims: List of Dimension instances defining the grid axes.
        map: Optional Zarr array for trace mapping, or None if not set.
        live_mask: Optional Zarr array indicating live traces, or None if not set.

    Notes:
        Computed attributes available after initialization:
        - `dim_names`: Tuple of dimension names.
        - `shape`: Tuple of dimension sizes.
        - `ndim`: Number of dimensions.

    Example:
        >>> from mdio.core import Dimension
        >>> dims = [Dimension(name="x", min=0, max=100, step=10)]
        >>> grid = Grid(dims)
        >>> grid.dim_names
        ('x',)
        >>> grid.shape
        (11,)
    """

    dims: list[Dimension]
    map: ZarrArray | None = None
    live_mask: ZarrArray | None = None

    _TARGET_MEMORY_PER_BATCH = 1 * 1024**3  # 1GB target for batch processing
    _INTERNAL_CHUNK_SIZE_TARGET = 10 * 1024**2  # 10MB target for chunks

    def __post_init__(self) -> None:
        """Initialize derived attributes."""
        self.dim_names = tuple(dim.name for dim in self.dims)
        self.shape = tuple(dim.size for dim in self.dims)
        self.ndim = len(self.dims)
        # Prepare attributes for lazy mapping; they will be set in build_map
        self.header_index_arrays: tuple[np.ndarray, ...] = ()
        self.num_traces: int = 0

    def __getitem__(self, item: int) -> Dimension:
        """Get a dimension by index."""
        return self.dims[item]

    def __setitem__(self, key: int, value: Dimension) -> None:
        """Set a dimension by index."""
        self.dims[key] = value

    def select_dim(self, name: str) -> Dimension:
        """Get a dimension by name."""
        index = self.dim_names.index(name)
        return self.dims[index]

    def get_min(self, name: str) -> float:
        """Get minimum value of a dimension by name."""
        return self.select_dim(name).min().item()

    def get_max(self, name: str) -> float:
        """Get maximum value of a dimension by name."""
        return self.select_dim(name).max().item()

    def serialize(self, stream_format: str) -> str:
        """Serialize the grid to a string buffer."""
        serializer = GridSerializer(stream_format)
        return serializer.serialize(self)

    @classmethod
    def deserialize(cls, stream: str, stream_format: str) -> Grid:
        """Deserialize a string buffer into a Grid instance."""
        serializer = GridSerializer(stream_format)
        return serializer.deserialize(stream)

    @classmethod
    def from_zarr(cls, zarr_root: zarr.Group) -> Grid:
        """Create a Grid instance from Zarr group attributes."""
        dims_list = zarr_root.attrs["dimension"]
        dims_list = [Dimension.from_dict(dim) for dim in dims_list]
        return cls(dims_list)

    def build_map(self, index_headers: HeaderArray) -> None:
        """Compute per-trace grid coordinates (lazy map).

        Instead of allocating a full `self.map` and `self.live_mask`, this computes, for each trace,
        its integer index along each dimension (excluding the sample dimension) and stores them in
        `self.header_index_arrays`. The full mapping can then be derived chunkwise when writing.

        Args:
            index_headers: Header array containing dimension indices (length = number of traces).
        """
        # Number of traces in the SEG-Y
        self.num_traces = int(index_headers.shape[0])

        # For each dimension except the final sample dimension, compute a 1D array of length
        # `num_traces` giving each trace's integer coordinate along that axis (via np.searchsorted).
        # Cast to uint32.
        idx_arrays: list[np.ndarray] = []
        for dim in self.dims[:-1]:
            hdr_vals = index_headers[dim.name]  # shape: (num_traces,)
            coords = np.searchsorted(dim, hdr_vals)  # integer indices
            coords = coords.astype(np.uint32)
            idx_arrays.append(coords)

        # Store as a tuple so that header_index_arrays[d][i] is "trace i's index along axis d"
        self.header_index_arrays = tuple(idx_arrays)

        # We no longer allocate `self.map` or `self.live_mask` here.
        # The full grid shape is `self.shape`, but mapping is done lazily per chunk.

    def get_traces_for_chunk(self, chunk_slices: tuple[slice, ...]) -> np.ndarray:
        """Return all trace IDs whose grid-coordinates fall inside the given chunk slices.

        Args:
            chunk_slices: Tuple of slice objects, one per grid dimension. For example,
                          (slice(i0, i1), slice(j0, j1), ...) corresponds to a single Zarr chunk
                          in index space (excluding the sample axis).

        Returns:
            A 1D NumPy array of trace indices (0-based) that lie within the hyper-rectangle defined
            by `chunk_slices`. If no traces fall in this chunk, returns an empty array.
        """
        # Initialize a boolean mask over all traces (shape: (num_traces,))
        mask = np.ones((self.num_traces,), dtype=bool)

        for dim_idx, sl in enumerate(chunk_slices):
            arr = self.header_index_arrays[dim_idx]  # shape: (num_traces,)
            start, stop = sl.start, sl.stop
            if start is not None:
                mask &= arr >= start
            if stop is not None:
                mask &= arr < stop
            if not mask.any():
                # No traces remain after this dimension's filtering
                return np.empty((0,), dtype=np.uint32)

        # Gather the trace IDs that survived all dimension tests
        return np.nonzero(mask)[0].astype(np.uint32)


class GridSerializer(Serializer):
    """Serializer implementation for Grid."""

    def serialize(self, grid: Grid) -> str:
        """Serialize Grid into buffer."""
        payload = [dim.to_dict() for dim in grid.dims]
        return self.serialize_func(payload)

    def deserialize(self, stream: str) -> Grid:
        """Deserialize buffer into Grid."""
        signature = inspect.signature(Grid)

        payload = self.deserialize_func(stream)
        payload = [Dimension.from_dict(dim) for dim in payload]
        payload = self.validate_payload({"dims": payload}, signature)

        return Grid(**payload)
