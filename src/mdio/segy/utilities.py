"""More utilities for reading SEG-Ys."""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

from dask.array.core import normalize_chunks

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

logger = logging.getLogger(__name__)


def find_trailing_ones_index(dim_blocks: tuple[int, ...]) -> int:
    """Finds the index where trailing '1's begin in a tuple of dimension block sizes.

    If all values are '1', returns 0.

    Args:
        dim_blocks: A list of integers representing the data chunk dimensions.

    Returns:
        The index indicating the breakpoint where the trailing sequence of "1s"
        begins, or `0` if all values in the list are `1`.

    Examples:
        >>> find_trailing_ones_index((7, 5, 1, 1))
        2

        >>> find_trailing_ones_index((1, 1, 1, 1))
        0
    """
    total_dims = len(dim_blocks)
    trailing_ones = itertools.takewhile(lambda x: x == 1, reversed(dim_blocks))
    trailing_ones_count = sum(1 for _ in trailing_ones)

    return total_dims - trailing_ones_count


# TODO (Dmitriy Repin): Investigate the following warning generated at test_3d_export
# https://github.com/TGSAI/mdio-python/issues/657
# "The specified chunks separate the stored chunks along dimension "inline" starting at index 256.
# This could degrade performance. Instead, consider rechunking after loading."
def segy_export_rechunker(
    chunks: dict[str, int],
    sizes: dict[str, int],
    dtype: DTypeLike,
    limit: str = "300M",
) -> dict[str, int]:
    """Determine chunk sizes for writing out SEG-Y given limit.

    This module finds the desired chunk sizes for given chunk size `limit` in a depth first order.

    On disk chunks for MDIO are mainly optimized for visualization and ML applications. When we
    want to do export back to SEG-Y, it makes sense to have larger virtual chunks for processing
    of traces. We also recursively merge multiple files to reduce memory footprint.

    We choose to adjust chunks to be approx. 300 MB. We also need to do this in the order of
    fastest changing axis to the slowest changing axis becase the traces are expected to be
    serialized in the natural data order.

    Args:
        chunks: The chunk sizes on disk, per dimension.
        sizes: Shape of the whole array, per dimension.
        dtype: Numpy `dtype` of the array.
        limit: Chunk size limit in, optional. Default is "300 MB"

    Returns:
        Adjusted chunk sizes for further processing
    """
    dim_names = list(sizes.keys())
    sample_dim_key = dim_names[-1]

    # set sample dim chunks (last one) to max
    prev_chunks = chunks.copy()
    prev_chunks[sample_dim_key] = sizes[sample_dim_key]

    new_chunks = {}
    for dim_name in reversed(list(prev_chunks)):
        tmp_chunks: dict[str, int | str] = prev_chunks.copy()
        tmp_chunks[dim_name] = "auto"

        new_chunks = normalize_chunks(
            chunks=tuple(tmp_chunks.values()),
            shape=tuple(sizes.values()),
            limit=limit,
            previous_chunks=tuple(prev_chunks.values()),
            dtype=dtype,
        )
        new_chunks = dict(zip(dim_names, new_chunks, strict=True))
        prev_chunks = new_chunks.copy()

    # Ensure the sample (last dim) is single chunk.
    new_chunks[sample_dim_key] = sizes[sample_dim_key]
    logger.debug("Auto export rechunking to: %s", new_chunks)
    return new_chunks
