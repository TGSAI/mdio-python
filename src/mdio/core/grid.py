"""Grid abstraction with serializers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import zarr

from mdio.constants import UINT32_MAX
from mdio.core import Dimension
from mdio.core.serialization import Serializer
from mdio.core.utils_write import get_constrained_chunksize

if TYPE_CHECKING:
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
        """Build trace mapping and live mask from header indices.

        Args:
            index_headers: Header array containing dimension indices.
        """
        # Determine data type for map based on grid size
        grid_size = np.prod(self.shape[:-1], dtype=np.uint64)
        map_dtype = np.uint64 if grid_size > UINT32_MAX else np.uint32
        fill_value = np.iinfo(map_dtype).max

        # Initialize Zarr arrays
        live_shape = self.shape[:-1]
        chunks = get_constrained_chunksize(
            shape=live_shape,
            dtype=map_dtype,
            max_bytes=self._INTERNAL_CHUNK_SIZE_TARGET,
        )
        self.map = zarr.full(live_shape, fill_value, dtype=map_dtype, chunks=chunks)
        self.live_mask = zarr.zeros(live_shape, dtype=bool, chunks=chunks)

        # Calculate batch size
        memory_per_trace_index = index_headers.itemsize
        batch_size = max(1, int(self._TARGET_MEMORY_PER_BATCH / memory_per_trace_index))
        total_live_traces = index_headers.size

        # Process headers in batches
        for start in range(0, total_live_traces, batch_size):
            end = min(start + batch_size, total_live_traces)
            live_dim_indices = []

            # Compute indices for the batch
            for dim in self.dims[:-1]:
                dim_hdr = index_headers[dim.name][start:end]
                indices = np.searchsorted(dim, dim_hdr).astype(np.uint32)
                live_dim_indices.append(indices)
            live_dim_indices = tuple(live_dim_indices)

            # Assign trace indices
            trace_indices = np.arange(start, end, dtype=np.uint64)
            self.map.vindex[live_dim_indices] = trace_indices
            self.live_mask.vindex[live_dim_indices] = True


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
