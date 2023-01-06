"""Indexing logic."""


import itertools
from math import ceil

import numpy as np
from zarr import Array


class ChunkIterator:
    """Chunk iterator multi-dimensional Zarr arrays.

    This iterator takes a zarr array and every time it is iterated, returns
    slices that align with chunk boundaries.

    Args:
        array: zarr.Array to get shape, and chunks from.
        chunk_samples: This is a flag to return the last dimension's
            slice as full, instead of chunks. Default is True.

    Attributes:
        arr_shape: Shape of the array.
        len_chunks: Length of chunks in each dimension.
        dim_chunks: Number of chunks in each dimension.
        num_chunks: Total number of chunks.
    """

    def __init__(self, array: Array, chunk_samples: bool = True):
        """Initialize iterator."""
        self.arr_shape = array.shape
        self.len_chunks = array.chunks

        # Handle the case when we don't want to slice the sample axis.
        if chunk_samples is False:
            self.len_chunks = self.len_chunks[:-1] + (self.arr_shape[-1],)

        # Compute number of chunks per dimension, and total number of chunks
        # TODO: Add strict=True and remove noqa when minimum Python is 3.10
        self.dim_chunks = [
            ceil(len_dim / chunk)
            for len_dim, chunk in zip(self.arr_shape, self.len_chunks)  # noqa: B905
        ]
        self.num_chunks = np.prod(self.dim_chunks)

        # Under the hood stuff for the iterator. This generates C-ordered
        # permutation of chunk indices.
        dim_ranges = [range(dim_len) for dim_len in self.dim_chunks]
        self._ranges = itertools.product(*dim_ranges)
        self._idx = 0

    def __iter__(self):
        """Iteration context."""
        return self

    def __len__(self):
        """Get total number of chunks."""
        return self.num_chunks

    def __next__(self):
        """Iteration logic."""
        if self._idx <= self.num_chunks:
            # We build slices here. It is dimension agnostic
            current_start = next(self._ranges)

            # TODO: Add strict=True and remove noqa when minimum Python is 3.10
            start_indices = tuple(
                dim * chunk
                for dim, chunk in zip(current_start, self.len_chunks)  # noqa: B905
            )

            # TODO: Add strict=True and remove noqa when minimum Python is 3.10
            stop_indices = tuple(
                (dim + 1) * chunk
                for dim, chunk in zip(current_start, self.len_chunks)  # noqa: B905
            )

            # TODO: Add strict=True and remove noqa when minimum Python is 3.10
            slices = tuple(
                slice(start, stop)
                for start, stop in zip(start_indices, stop_indices)  # noqa: B905
            )

            self._idx += 1

            return slices

        else:
            raise StopIteration
