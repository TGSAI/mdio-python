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
from mdio.segy.parsers import parse_headers

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from segy import SegyFile
    from segy.arrays import HeaderArray

    from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate


logger = logging.getLogger(__name__)


def get_grid_plan(  # noqa:  C901
    segy_file: SegyFile,
    chunksize: tuple[int, ...] | None,
    template: AbstractDatasetTemplate,
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
        template: MDIO template where coordinate names and domain will be taken.
        return_headers: Option to return parsed headers with `Dimension` objects. Default is False.
        grid_overrides: Option to add grid overrides. See main documentation.

    Returns:
        All index dimensions and chunksize or dimensions and chunksize together with header values.
    """
    if grid_overrides is None:
        grid_overrides = {}

    # Keep only dimension and non-dimension coordinates excluding the vertical axis
    horizontal_dimensions = template.dimension_names[:-1]
    horizontal_coordinates = horizontal_dimensions + template.coordinate_names
    headers_subset = parse_headers(segy_file=segy_file, subset=horizontal_coordinates)

    # Handle grid overrides.
    override_handler = GridOverrider()
    headers_subset, horizontal_coordinates, chunksize = override_handler.run(
        headers_subset,
        horizontal_coordinates,
        chunksize=chunksize,
        grid_overrides=grid_overrides,
    )

    dimensions = []
    for dim_name in horizontal_dimensions:
        dim_unique = np.unique(headers_subset[dim_name])
        dimensions.append(Dimension(coords=dim_unique, name=dim_name))

    sample_labels = segy_file.sample_labels / 1000  # normalize

    if all(sample_labels.astype("int64") == sample_labels):
        sample_labels = sample_labels.astype("int64")

    vertical_dim = Dimension(coords=sample_labels, name=template.trace_domain)
    dimensions.append(vertical_dim)

    if return_headers:
        return dimensions, chunksize, headers_subset

    return dimensions, chunksize


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
