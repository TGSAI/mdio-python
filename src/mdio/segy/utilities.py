"""More utilities for reading SEG-Ys."""


from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
from dask.array.core import auto_chunks

from mdio.core import Dimension
from mdio.segy.parsers import parse_sample_axis
from mdio.segy.parsers import parse_trace_headers


def get_grid_plan(
    segy_path: str,
    segy_endian: str,
    index_bytes: Sequence[int],
    index_names: Sequence[str],
    index_lengths: Sequence[int],
    binary_header: dict,
    return_headers: bool = False,
) -> list[Dimension] | tuple[list[Dimension], npt.ArrayLike]:
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
        index_lengths: Tuple of the byte lengths for the index attributes.
            Default will be 4-byte for all indices.
        binary_header: Dictionary containing binary header key, value pairs.
        return_headers: Option to return parsed headers with `Dimension` objects.
            Default is False.

    Returns:
        All index dimensions or dimensions together with header values.
    """
    index_dim = len(index_bytes)

    # Default is 4-byte for each index.
    index_lengths = [4] * index_dim if index_lengths is None else index_lengths

    index_headers = parse_trace_headers(
        segy_path=segy_path,
        segy_endian=segy_endian,
        byte_locs=index_bytes,
        byte_lengths=index_lengths,
    )

    if index_names is None:
        index_names = [f"index_{dim}" for dim in range(index_dim)]

    dims = []
    for dim, dim_name in enumerate(index_names):
        dim_unique = np.unique(index_headers[:, dim])
        dims.append(Dimension(coords=dim_unique, name=dim_name))

    sample_dim = parse_sample_axis(binary_header=binary_header)

    dims.append(sample_dim)

    return dims, index_headers if return_headers else dims


def segy_export_rechunker(
    chunks: tuple[int],
    shape: tuple[int],
    dtype: npt.DTypeLike,
    limit: str = "300M",
) -> tuple[int]:
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
