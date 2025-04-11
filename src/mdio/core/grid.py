"""Grid abstraction with serializers."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import zarr
from dask.array.core import normalize_chunks
from dask.array.rechunk import _balance_chunksizes

from mdio.constants import INT32_MAX
from mdio.constants import UINT32_MAX
from mdio.constants import UINT64_MAX
from mdio.core import Dimension
from mdio.core.serialization import Serializer


@dataclass
class Grid:
    """N-Dimensional grid class.

    This grid object holds information about bounds and
    increments of an N-Dimensional grid.

    The dimensions must be provided as supported MDIO dimension
    objects. They can be found in `mdio.core.dimension` module.

    Args:
        dims: List of dimension instances.

    """

    dims: list[Dimension]

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

    def build_map(self, index_headers):
        """Build a map for live traces based on `index_headers`.

        Args:
            index_headers: Headers to be normalized (indexed)
        """
        live_dim_indices = tuple()
        for dim in self.dims[:-1]:
            dim_hdr = index_headers[dim.name]
            live_dim_indices += (np.searchsorted(dim, dim_hdr),)

        # There were cases where ingestion would overflow a signed int32.
        # It's unlikely that we overflow the uint32_max, but this helps
        # prevent any issues while keeping the memory footprint as low as possible.
        grid_size = np.prod(self.shape[:-1])
        if grid_size > UINT32_MAX - 1:
            # We use UINT32_MAX-1 to ensure that the assumption below is not violated.
            # "far away" is relative.
            logging.warning(
                f"Grid size {grid_size} exceeds UINT32_MAX ({UINT32_MAX - 1}). "
                "Using uint64 for trace map which will use more memory."
            )
            dtype = "uint64"
            fill_value = UINT64_MAX
        else:
            dtype = "uint32"
            fill_value = UINT32_MAX

        # We set dead traces to max uint32/uint64 value.
        # Should be far away from actual trace counts.
        self.map = zarr.full(self.shape[:-1], dtype=dtype, fill_value=fill_value)
        self.map.vindex[live_dim_indices] = range(len(live_dim_indices[0]))

        self.live_mask = zarr.zeros(self.shape[:-1], dtype="bool")
        self.live_mask.vindex[live_dim_indices] = 1


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


class _EmptyGrid:
    """Empty volume for Grid mocking."""

    def __init__(self, shape: Sequence[int], dtype: np.dtype = np.bool):
        """Initialize the empty grid."""
        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, item):
        """Get item from the empty grid."""
        return self.dtype.type(0)


def _calculate_live_mask_chunksize(grid: Grid) -> Sequence[int]:
    """Calculate the optimal chunksize for the live mask.

    Args:
        grid: The grid to calculate the chunksize for.

    Returns:
        A sequence of integers representing the optimal chunk size for each dimension
        of the grid.
    """
    try:
        return _calculate_optimal_chunksize(grid.live_mask, INT32_MAX // 4)
    except AttributeError:
        # Create an empty array with the same shape and dtype as the live mask would have
        return _calculate_optimal_chunksize(_EmptyGrid(grid.shape[:-1]), INT32_MAX // 4)


def _calculate_optimal_chunksize(  # noqa: C901
    volume: np.ndarray | zarr.Array, max_bytes: int
) -> Sequence[int]:
    """Calculate the optimal chunksize for an N-dimensional data volume.

    Args:
        volume: The volume to calculate the chunksize for.
        max_bytes: The maximum allowed number of bytes per chunk.

    Returns:
        A sequence of integers representing the optimal chunk size for each dimension
        of the grid.
    """
    shape = volume.shape
    chunks = normalize_chunks(
        "auto",
        shape,
        dtype=volume.dtype,
        limit=max_bytes,
    )
    return tuple(_balance_chunksizes(chunk)[0] for chunk in chunks)
