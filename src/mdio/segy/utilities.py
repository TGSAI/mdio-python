"""More utilities for reading SEG-Ys."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from dask.array.core import auto_chunks
from numpy.typing import DTypeLike
from numpy.typing import NDArray

from mdio.core import Dimension
from mdio.segy.byte_utils import Dtype
from mdio.segy.geometry import GridOverrider
from mdio.segy.parsers import parse_sample_axis
from mdio.segy.parsers import parse_trace_headers


def get_grid_plan(  # noqa:  C901
    segy_path: str,
    segy_endian: str,
    index_bytes: Sequence[int],
    index_names: Sequence[str],
    index_types: Sequence[Dtype],
    chunksize: Sequence[int],
    binary_header: dict,
    return_headers: bool = False,
    grid_overrides: dict | None = None,
) -> (
    tuple[list[Dimension], tuple[int]]
    | tuple[list[Dimension], tuple[int], dict[str, NDArray]]
):
    """Infer dimension ranges, and increments.

    Generates multiple dimensions with the following steps:
    1. Read index headers
    2. Get min, max, and increments
    3. Create `Dimension` with appropriate range, index, and description.
    4. Create `Dimension` for sample axis using binary header.

    Args:
        segy_path: Path to the input SEG-Y file
        segy_endian: Endianness of the input SEG-Y.
        index_bytes: Tuple of the byte location for the index attributes
        index_names: Tuple of the names for the index attributes
        index_types: Tuple of the data types for the index attributes.
        chunksize:  Chunk sizes to be used in grid plan.
        binary_header: Dictionary containing binary header key, value pairs.
        return_headers: Option to return parsed headers with `Dimension` objects.
            Default is False.
        grid_overrides: Option to add grid overrides. See main documentation.

    Returns:
        All index dimensions and chunksize or dimensions and chunksize together
            with header values.
    """
    if grid_overrides is None:
        grid_overrides = {}

    index_dim = len(index_bytes)

    if index_names is None:
        index_names = [f"index_{dim}" for dim in range(index_dim)]

    index_headers = parse_trace_headers(
        segy_path=segy_path,
        segy_endian=segy_endian,
        byte_locs=index_bytes,
        byte_types=index_types,
        index_names=index_names,
    )

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

    sample_dim = parse_sample_axis(binary_header=binary_header)

    dims.append(sample_dim)

    if return_headers:
        return dims, chunksize, index_headers

    return dims, chunksize


def segy_export_rechunker(
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
    dtype: DTypeLike,
    limit: str = "300M",
) -> tuple[int, ...]:
    """Determine chunk sizes for writing out SEG-Y given limit.

    This module finds the desired chunk sizes for given chunk size
    `limit` in a depth first order.

    On disk chunks for MDIO are mainly optimized for visualization
    and ML applications. When we want to do export back to SEG-Y, it
    makes sense to have larger virtual chunks for processing of traces.
    We also recursively merge multiple files to reduce memory footprint.

    We choose to adjust chunks to be approx. 300 MB. We also need to do
    this in the order of fastest changing axis to the slowest changing
    axis becase the traces are expected to be serialized in the natural
    data order.

    Args:
        chunks: The chunk sizes on disk.
        shape: Shape of the whole array.
        dtype: Numpy `dtype` of the array.
        limit: Chunk size limit in, optional. Default is "300 MB"

    Returns:
        Adjusted chunk sizes for further processing

    Raises:
        ValueError: If resulting chunks will split file on disk.
    """
    ndim = len(shape) - 1  # minus the sample axis

    # set sample chunks to max
    prev_chunks = chunks[:-1] + (shape[-1],)

    for idx in range(ndim, -1, -1):
        tmp_chunks = prev_chunks[:idx] + ("auto",) + prev_chunks[idx + 1 :]

        new_chunks = auto_chunks(
            chunks=tmp_chunks,
            shape=shape,
            limit=limit,
            previous_chunks=prev_chunks,
            dtype=dtype,
        )

        # Ensure it is integers
        new_chunks = tuple(map(int, new_chunks))
        prev_chunks = new_chunks

    # TODO: Add strict=True and remove noqa when minimum Python is 3.10
    qc_iterator = zip(new_chunks, chunks, shape)  # noqa: B905

    for idx, (dim_new_chunk, dim_chunk, dim_size) in enumerate(qc_iterator):
        # Sometimes dim_chunk can be larger than dim_size. This catches when
        # that is False and the new chunk will be smaller than original
        if dim_new_chunk < dim_chunk < dim_size:
            msg = (
                f"Dimension {idx} chunk size in {new_chunks=} is smaller than "
                f"the disk {chunks=} with given {limit=}. This will cause very "
                f"poor performance due to redundant reads. Please increase limit "
                f"to get larger chunks. However, this may require more memory."
            )
            raise ValueError(msg)

    return new_chunks
