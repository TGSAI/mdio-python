"""More utilities for reading SEG-Ys."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Sequence

import numpy as np
import numpy.typing as npt
from dask.array.core import auto_chunks

from mdio.core import Dimension
from mdio.segy.byte_utils import Dtype
from mdio.segy.parsers import parse_sample_axis
from mdio.segy.parsers import parse_trace_headers


logger = logging.getLogger(__name__)


class GeometryTemplateType(Enum):
    """Geometry template types as enum."""

    STREAMER_A = 1
    STREAMER_B = 2


r"""
STREAMER_A
==========
Cable 1 ->          1------------------20
Cable 2 ->         1-----------------20
.                 1-----------------20
.          ⛴ ☆  1-----------------20
.                 1-----------------20
Cable 6 ->         1-----------------20
Cable 7 ->          1-----------------20


STREAMER_B
==========
Cable 1 ->          1------------------20
Cable 2 ->         21-----------------40
.                 41-----------------60
.          ⛴ ☆  61-----------------80
.                 81----------------100
Cable 6 ->         101---------------120
Cable 7 ->          121---------------140

"""


def get_grid_plan(  # noqa:  C901
    segy_path: str,
    segy_endian: str,
    index_bytes: Sequence[int],
    index_names: Sequence[str],
    index_types: Sequence[Dtype],
    binary_header: dict,
    return_headers: bool = False,
    grid_overrides: dict | None = None,
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
        index_types: Tuple of the data types for the index attributes.
        binary_header: Dictionary containing binary header key, value pairs.
        return_headers: Option to return parsed headers with `Dimension` objects.
            Default is False.
        grid_overrides: Option to add grid overrides. See main documentation.

    Returns:
        All index dimensions or dimensions together with header values.

    Raises:
        ValueError: If appropriate grid override parameters are not provided.
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

    if "AutoChannelWrap" in grid_overrides:
        trace_qc_count = None
        cable_idx = index_names.index("cable")
        chan_idx = index_names.index("channel")
        if "AutoChannelTraceQC" in grid_overrides:
            trace_qc_count = int(grid_overrides["AutoChannelTraceQC"])
        unique_cables, cable_chan_min, _cable_chan_max, geom_type = qc_index_headers(
            index_headers, index_names, trace_qc_count
        )

        logger.info(f"Ingesting dataset as {geom_type.name}")
        # TODO: Add strict=True and remove noqa when min Python is 3.10
        for cable, chan_min, chan_max in zip(  # noqa: B905
            unique_cables, cable_chan_min, _cable_chan_max
        ):
            logger.info(
                f"Cable: {cable} has min chan: {chan_min} and max chan: {chan_max}"
            )

        # This might be slow and potentially could be improved with a rewrite
        # to prevent so many lookups
        if geom_type == GeometryTemplateType.STREAMER_B:
            for idx, cable in enumerate(unique_cables):
                cable_idxs = np.where(index_headers["cable"][:] == cable)
                cc_min = cable_chan_min[idx]
                # print(f"idx = {idx}  cable = {cable} cc_min={cc_min}")
                index_headers["channel"][cable_idxs] = (
                    index_headers["channel"][cable_idxs] - cc_min + 1
                )

    if "CalculateCable" in grid_overrides:
        if "ChannelsPerCable" in grid_overrides:
            channels_per_cable = grid_overrides["ChannelsPerCable"]
            index_headers["cable"] = (
                index_headers["channel"] - 1
            ) // channels_per_cable + 1
        else:
            raise ValueError("'ChannelsPerCable' must be specified to calculate cable.")

    if "ChannelWrap" in grid_overrides:
        if grid_overrides["ChannelWrap"] is True:
            if "ChannelsPerCable" in grid_overrides:
                channels_per_cable = grid_overrides["ChannelsPerCable"]
                index_headers["channel"] = (
                    index_headers["channel"] - 1
                ) % channels_per_cable + 1
        else:
            raise ValueError("'ChannelsPerCable' must be specified to wrap channels.")

    for index_name in index_names:
        dim_unique = np.unique(index_headers[index_name])
        dims.append(Dimension(coords=dim_unique, name=index_name))

    sample_dim = parse_sample_axis(binary_header=binary_header)

    dims.append(sample_dim)

    return dims, index_headers if return_headers else dims


def qc_index_headers(
    index_headers: npt.ArrayLike,
    index_names: Sequence[str],
    trace_qc_count: int | None = None,
) -> tuple(npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, GeometryTemplateType) | None:
    """Check input headers for segy input to help determin geometry.

    This function reads in trace_qc_count headers and finds the unique cable values.
    The function then checks to make sure channel numbers for different cables do
    not overlap.

    Args:
        index_headers: numpy array with index headers
        index_names: Tuple of the names for the index attributes
        trace_qc_count: Number of traces to use in QC (default all)

    Returns:
        None: if not shot, cable, channel
        if shot, cable, channels:
            unique_cables: Array with the unique cable ids
            cable_chan_min: Array containing the min channel number for each cable,
            cable_chan_max: Array containing the max channel number for each cable,
            geom_type:  Whether type a or b (wrapped or unwrapped chans)
    """
    if "cable" in index_names and "channel" in index_names and "shot" in index_names:
        if trace_qc_count is None:
            trace_qc_count = index_headers["cable"].shape[0]
        if trace_qc_count > index_headers["cable"].shape[0]:
            trace_qc_count = index_headers["cable"].shape[0]

        # Find unique cable ids
        unique_cables = np.sort(np.unique(index_headers["cable"][0:trace_qc_count]))

        # Find channel min and max values for each cable
        cable_chan_min = np.empty(unique_cables.shape)
        cable_chan_max = np.empty(unique_cables.shape)

        for idx, cable in enumerate(unique_cables):
            my_chan = np.take(
                index_headers["channel"][0:trace_qc_count],
                np.where(index_headers["cable"][0:trace_qc_count] == cable),
            )
            cable_chan_min[idx] = np.min(my_chan)
            cable_chan_max[idx] = np.max(my_chan)

        # Check channel numbers do not overlap for case B
        geom_type = GeometryTemplateType.STREAMER_B
        for idx, cable in enumerate(unique_cables):
            min_val = cable_chan_min[idx]
            max_val = cable_chan_max[idx]
            for idx2, cable2 in enumerate(unique_cables):
                if (
                    cable_chan_min[idx2] < max_val
                    and cable_chan_max[idx2] > min_val
                    and (cable2 != cable)
                ):
                    geom_type = GeometryTemplateType.STREAMER_A

        # Return cable_chan_min values
        return unique_cables, cable_chan_min, cable_chan_max, geom_type


def segy_export_rechunker(
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
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
