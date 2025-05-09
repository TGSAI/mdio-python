"""More utilities for reading SEG-Ys."""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from dask.array.core import normalize_chunks

from mdio.core import Dimension
from mdio.segy.geometry import GridOverrider
from mdio.segy.parsers import parse_index_headers

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from segy import SegyFile
    from segy.arrays import HeaderArray


logger = logging.getLogger(__name__)


def get_grid_plan(  # noqa:  C901
    segy_file: SegyFile,
    chunksize: list[int],
    return_headers: bool = False,
    grid_overrides: dict[str, Any] | None = None,
) -> tuple[list[Dimension], tuple[int, ...]] | tuple[list[Dimension], tuple[int, ...], HeaderArray]:
    """Infer dimension ranges, and increments.

    Generates multiple dimensions with the following steps:
    1. Read index headers
    2. Get min, max, and increments
    3. Create `Dimension` with appropriate range, index, and description.
    4. Create `Dimension` for sample axis using binary header.

    Args:
        segy_file: SegyFile instance.
        chunksize:  Chunk sizes to be used in grid plan.
        return_headers: Option to return parsed headers with `Dimension` objects. Default is False.
        grid_overrides: Option to add grid overrides. See main documentation.

    Returns:
        All index dimensions and chunksize or dimensions and chunksize together with header values.
    """
    if grid_overrides is None:
        grid_overrides = {}

    index_headers = parse_index_headers(segy_file=segy_file)
    index_names = list(index_headers.dtype.names)

    dims = []

    # Handle grid overrides.
    override_handler = GridOverrider()
    index_headers, index_names, chunksize = override_handler.run(
        index_headers,
        index_names,
        chunksize=chunksize,
        grid_overrides=grid_overrides,
    )

    for index_name in index_names:
        dim_unique = np.unique(index_headers[index_name])
        dims.append(Dimension(coords=dim_unique, name=index_name))

    sample_labels = segy_file.sample_labels / 1000  # normalize

    if all(sample_labels.astype("int64") == sample_labels):
        sample_labels = sample_labels.astype("int64")

    sample_dim = Dimension(coords=sample_labels, name="sample")

    dims.append(sample_dim)

    if return_headers:
        return dims, chunksize, index_headers

    return dims, chunksize


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


def segy_export_rechunker(
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
    dtype: DTypeLike,
    limit: str = "300M",
) -> tuple[tuple[int, ...], ...]:
    """Determine chunk sizes for writing out SEG-Y given limit.

    This module finds the desired chunk sizes for given chunk size `limit` in a depth first order.

    On disk chunks for MDIO are mainly optimized for visualization and ML applications. When we
    want to do export back to SEG-Y, it makes sense to have larger virtual chunks for processing
    of traces. We also recursively merge multiple files to reduce memory footprint.

    We choose to adjust chunks to be approx. 300 MB. We also need to do this in the order of
    fastest changing axis to the slowest changing axis becase the traces are expected to be
    serialized in the natural data order.

    Args:
        chunks: The chunk sizes on disk.
        shape: Shape of the whole array.
        dtype: Numpy `dtype` of the array.
        limit: Chunk size limit in, optional. Default is "300 MB"

    Returns:
        Adjusted chunk sizes for further processing
    """
    ndim = len(shape) - 1  # minus the sample axis

    # set sample chunks to max
    prev_chunks = chunks[:-1] + (shape[-1],)

    new_chunks = ()
    for idx in range(ndim, -1, -1):
        tmp_chunks = prev_chunks[:idx] + ("auto",) + prev_chunks[idx + 1 :]

        new_chunks = normalize_chunks(
            chunks=tmp_chunks,
            shape=shape,
            limit=limit,
            previous_chunks=prev_chunks,
            dtype=dtype,
        )

        prev_chunks = new_chunks

    # Ensure the sample (last dim) is single chunk.
    if len(new_chunks[-1]) != 1:
        new_chunks = new_chunks[:-1] + (shape[-1],)

    logger.debug("Auto export rechunking to: %s", new_chunks)
    return new_chunks
