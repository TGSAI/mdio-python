"""Indexing logic."""

import itertools
from math import ceil

import numpy as np
from zarr import Array


class ChunkIterator:
    """Iterator for traversing a Zarr array in chunks.

    This iterator yields tuples of slices corresponding to the chunk boundaries of a Zarr array.
    It supports chunking all dimensions or taking the full extent of the last dimension.

    Args:
        array: The Zarr array to iterate, providing shape and chunk sizes.
        chunk_samples: If True, chunks all dimensions. If False, takes the full extent of the
            last dimension. Defaults to True.


    Example:
        >>> import zarr
        >>> arr = zarr.array(np.zeros((10, 20)), chunks=(3, 4))
        >>> it = ChunkIterator(arr)
        >>> for slices in it:
        ...     print(slices)
        (slice(0, 3, None), slice(0, 4, None))
        (slice(0, 3, None), slice(4, 8, None))
        ...
        >>> it = ChunkIterator(arr, chunk_samples=False)
        >>> for slices in it:
        ...     print(slices)
        (slice(0, 3, None), slice(0, 20, None))
        (slice(3, 6, None), slice(0, 20, None))
        ...
    """

    def __init__(self, array: Array, chunk_samples: bool = True):
        self.arr_shape = array.shape
        self.len_chunks = array.chunks

        # If chunk_samples is False, set the last dimension's chunk size to its full extent
        if not chunk_samples:
            self.len_chunks = self.len_chunks[:-1] + (self.arr_shape[-1],)

        # Calculate the number of chunks per dimension
        self.dim_chunks = [
            ceil(len_dim / chunk)
            for len_dim, chunk in zip(self.arr_shape, self.len_chunks, strict=True)
        ]
        self.num_chunks = np.prod(self.dim_chunks)

        # Set up chunk index combinations using ranges for each dimension
        dim_ranges = [range(dim_len) for dim_len in self.dim_chunks]
        self._ranges = itertools.product(*dim_ranges)
        self._idx = 0

    def __iter__(self) -> "ChunkIterator":
        """Return the iterator object itself."""
        return self

    def __len__(self) -> int:
        """Return the total number of chunks."""
        return self.num_chunks

    def __next__(self) -> tuple[slice, ...]:
        """Yield the next set of chunk slices.

        Returns:
            A tuple of slice objects for each dimension.

        Raises:
            StopIteration: When all chunks have been iterated over.
        """
        if self._idx < self.num_chunks:
            current_start = next(self._ranges)

            start_indices = tuple(
                dim * chunk for dim, chunk in zip(current_start, self.len_chunks, strict=True)
            )

            stop_indices = tuple(
                (dim + 1) * chunk for dim, chunk in zip(current_start, self.len_chunks, strict=True)
            )

            slices = tuple(
                slice(start, stop) for start, stop in zip(start_indices, stop_indices, strict=True)
            )

            self._idx += 1
            return slices

        raise StopIteration
