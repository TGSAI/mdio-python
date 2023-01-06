"""Grid abstraction with serializers."""


from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import zarr

from mdio.constants import UINT32_MAX
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

        # TODO: Add strict=True and remove noqa when minimum Python is 3.10
        for dim, dim_hdr in zip(self.dims, index_headers.T):  # noqa: B905
            live_dim_indices += (np.searchsorted(dim, dim_hdr),)

        # We set dead traces to uint32 max. Should be far away from actual trace counts.
        self.map = zarr.full(self.shape[:-1], dtype="uint32", fill_value=UINT32_MAX)
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
