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
    """N-Dimensional grid class.

    This grid object holds information about bounds and
    increments of an N-Dimensional grid.

    The dimensions must be provided as supported MDIO dimension
    objects. They can be found in `mdio.core.dimension` module.

    Args:
        dims: List of dimension instances.

    Attributes:
        dims: List of dimension instances.
    """

    dims: list[Dimension]
    map: ZarrArray | None = None
    live_mask: ZarrArray | None = None

    _TARGET_MEMORY_PER_BATCH = 1 * 1024**3  # 1GB target for batch process map
    _INTERNAL_CHUNK_SIZE_TARGET = 10 * 1024**2  # 10MB target for internal chunks

    def __post_init__(self):
        """Initialize convenience properties."""
        self.dim_names = tuple(dim.name for dim in self.dims)
        self.shape = tuple(dim.size for dim in self.dims)
        self.ndim = len(self.dims)

    def __getitem__(self, item) -> Dimension:
        """Gets a specific dimension by index."""
        return self.dims[item]

    def __setitem__(self, key, value: Dimension) -> None:
        """Sets a specific dimension by index."""
        self.dims[key] = value

    def select_dim(self, name) -> Dimension:
        """Gets a specific dimension by name."""
        index = self.dim_names.index(name)
        return self.dims[index]

    def get_min(self, name):
        """Get minimum value of a dimension with a given name."""
        return self.select_dim(name).min()

    def get_max(self, name):
        """Get maximum value of a dimension with a given name."""
        return self.select_dim(name).max()

    def serialize(self, stream_format):
        """Serialize the Grid into buffer."""
        serializer = GridSerializer(stream_format)
        return serializer.serialize(self)

    @classmethod
    def deserialize(cls, stream, stream_format):
        """Deserialize buffer into Grid."""
        serializer = GridSerializer(stream_format)
        return serializer.deserialize(stream)

    # TODO: Make this a deserialize option
    @classmethod
    def from_zarr(cls, zarr_root: zarr.Group):
        """Deserialize grid from Zarr attributes."""
        dims_list = zarr_root.attrs["dimension"]
        dims_list = [Dimension.from_dict(dim) for dim in dims_list]

        return cls(dims_list)

    def build_map(self, index_headers: HeaderArray) -> None:
        """Build a map for live traces based on `index_headers`.

        Args:
            index_headers: Headers to be normalized (indexed)
        """
        # Determine data type for the map based on grid size
        grid_size = np.prod(self.shape[:-1])
        map_dtype = "uint64" if grid_size > UINT32_MAX else "uint32"
        fill_value = np.iinfo(map_dtype).max

        # Initialize Zarr arrays for the map and live mask
        live_shape = self.shape[:-1]
        chunks = get_constrained_chunksize(
            shape=live_shape,
            dtype=map_dtype,
            max_bytes=self._INTERNAL_CHUNK_SIZE_TARGET,
        )
        # Temporary zarrs for ingestion.
        self.map = zarr.full(live_shape, fill_value, dtype=map_dtype, chunks=chunks)
        self.live_mask = zarr.zeros(live_shape, dtype="bool", chunks=chunks)

        # Calculate batch size for processing
        memory_per_trace_index = index_headers.itemsize
        batch_size = int(self._TARGET_MEMORY_PER_BATCH / memory_per_trace_index)
        total_live_traces = index_headers.size

        # Process live traces in batches
        for start in range(0, total_live_traces, batch_size):
            end = min(start + batch_size, total_live_traces)

            # Compute indices for the current batch
            live_dim_indices = []
            for dim in self.dims[:-1]:
                dim_hdr = index_headers[dim.name][start:end]
                indices = np.searchsorted(dim, dim_hdr).astype(np.uint32)
                live_dim_indices.append(indices)
            live_dim_indices = tuple(live_dim_indices)

            # Generate trace indices for the batch
            trace_indices = np.arange(start, end, dtype=np.uint64)

            # Update Zarr arrays for the batch
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
        payload = dict(dims=payload)
        payload = self.validate_payload(payload, signature)

        return Grid(**payload)
