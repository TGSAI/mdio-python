"""Indexing logic."""

import itertools
from math import ceil

import numpy as np


class ChunkIterator:
    """Chunk iterator for multi-dimensional arrays.

    This iterator takes an array shape and chunks and every time it is iterated, it returns
    a dictionary (if dimensions are provided) or a tuple of slices that align with
    chunk boundaries. When dimensions are provided, they are used as the dictionary keys.

    Args:
        shape: The shape of the array.
        chunks: The chunk sizes for each dimension.
        dim_names: The names of the array dimensions, to be used with DataArray.isel().
                   If the dim_names are not provided, a tuple of the slices will be returned.

    Attributes:             # noqa: DOC602
        arr_shape: Shape of the array.
        len_chunks: Length of chunks in each dimension.
        dim_chunks: Number of chunks in each dimension.
        num_chunks: Total number of chunks.

    Examples:
        >> chunks = (3, 4, 5)
        >> shape = (5, 11, 19)
        >> dims = ["inline", "crossline", "depth"]
        >>
        >> iter = ChunkIterator(shape=shape, chunks=chunks, dim_names=dims)
        >> for i in range(13):
        >>    region = iter.__next__()
        >> print(region)
        { "inline": slice(3,6, None), "crossline": slice(0,4, None), "depth": slice(0,5, None) }

        >> iter = ChunkIterator(shape=shape, chunks=chunks, dim_names=None)
        >> for i in range(13):
        >>    region = iter.__next__()
        >> print(region)
        (slice(3,6,None), slice(0,4,None), slice(0,5,None))
    """

    def __init__(self, shape: tuple[int, ...], chunks: tuple[int, ...], dim_names: tuple[str, ...] = None):
        self.arr_shape = tuple(shape)  # Deep copy to ensure immutability
        self.len_chunks = tuple(chunks)  # Deep copy to ensure immutability
        self.dims = dim_names

        # Compute number of chunks per dimension, and total number of chunks
        self.dim_chunks = tuple(
            [ceil(len_dim / chunk) for len_dim, chunk in zip(self.arr_shape, self.len_chunks, strict=True)]
        )
        self.num_chunks = np.prod(self.dim_chunks)

        # Under the hood stuff for the iterator. This generates C-ordered
        # permutation of chunk indices.
        dim_ranges = [range(dim_len) for dim_len in self.dim_chunks]
        self._ranges = itertools.product(*dim_ranges)
        self._idx = 0

    def __iter__(self) -> "ChunkIterator":
        """Iteration context."""
        return self

    def __len__(self) -> int:
        """Get total number of chunks."""
        return self.num_chunks

    def __next__(self) -> dict[str, slice]:
        """Iteration logic."""
        if self._idx <= self.num_chunks:
            # We build slices here. It is dimension agnostic
            current_start = next(self._ranges)

            start_indices = tuple(dim * chunk for dim, chunk in zip(current_start, self.len_chunks, strict=True))

            # Calculate stop indices, making the last slice fit the data exactly
            stop_indices = tuple(
                min((dim + 1) * chunk, self.arr_shape[i])
                for i, (dim, chunk) in enumerate(zip(current_start, self.len_chunks, strict=True))
            )

            slices = tuple(slice(start, stop) for start, stop in zip(start_indices, stop_indices, strict=True))

            if self.dims:  # noqa SIM108
                # Example
                # {"inline":slice(3,6,None), "crossline":slice(0,4,None), "depth":slice(0,5,None)}
                result = dict(zip(self.dims, slices, strict=False))
            else:
                # Example
                # (slice(3,6,None), slice(0,4,None), slice(0,5,None))
                result = slices

            self._idx += 1

            return result

        raise StopIteration
