"""More utilities for reading SEG-Ys."""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np
from dask.array.core import normalize_chunks
from segy.schema import ScalarType as SegyScalarType

from mdio.builder.schemas.dtype import ScalarType as MdioScalarType
from mdio.builder.schemas.dtype import StructuredType
from mdio.converters.type_converter import to_numpy_dtype
from mdio.converters.type_converter import to_structured_type

if TYPE_CHECKING:
    from dask.array import Array as DaskArray
    from numpy.typing import DTypeLike
    from numpy.typing import NDArray
    from segy.schema import SegySpec


logger = logging.getLogger(__name__)


def ibm32_header_field_names(segy_spec: SegySpec) -> set[str]:
    """Return the names of trace-header fields declared as IBM 32-bit floats.

    The segy schema maps an ``ibm32`` header field to a raw ``uint32`` slot (the 4-byte
    IBM word) but decodes it to ``float32`` on read. Callers use these names to promote
    the affected fields to ``float32`` so the decoded value is stored and projected
    without truncating decimals or wrapping the sign of negative values.

    Args:
        segy_spec: SEG-Y specification whose trace header fields are inspected.

    Returns:
        Set of header field names whose declared format is ``ibm32``.
    """
    return {field.name for field in segy_spec.trace.header.fields if field.format == SegyScalarType.IBM32}


def build_mdio_header_type(segy_spec: SegySpec) -> StructuredType:
    """Build the MDIO ``headers`` variable type from a SegySpec.

    ``ibm32`` header fields are promoted from their raw ``uint32`` slot to ``float32`` so
    the persisted header matches the decoded array the ingestion worker writes. Without
    this promotion the decoded float would be cast down to an integer on write, truncating
    decimals (``118.625`` -> ``118``) and wrapping signed values (``-50.25`` -> a large
    unsigned integer).

    Args:
        segy_spec: SEG-Y specification describing the trace header layout.

    Returns:
        The MDIO structured type for the persisted ``headers`` variable.
    """
    structured = to_structured_type(segy_spec.trace.header.dtype)
    ibm32_names = ibm32_header_field_names(segy_spec)
    for field in structured.fields:
        if field.name in ibm32_names:
            field.format = MdioScalarType.FLOAT32
    return structured


def project_headers_to_segy_spec(headers: DaskArray, segy_spec: SegySpec) -> DaskArray:
    """Project stored MDIO trace headers onto the SegySpec trace header layout.

    ``SegyFactory.create_traces`` assigns headers by numpy structured-array slot position,
    not by field name, so the input must expose exactly the SegySpec fields in SegySpec
    order. A packed (no-padding), native-byte-order dtype is used to avoid numpy byteswap
    artifacts over padding bytes.

    Args:
        headers: Dask array holding MDIO trace headers with structured dtype.
        segy_spec: Target SegySpec describing the output SEG-Y trace header layout.

    Returns:
        Dask array with a packed, native-byte-order dtype ordered like the SegySpec.

    Raises:
        ValueError: If SegySpec requests header fields that do not exist in MDIO headers.
    """
    spec_header_dtype = segy_spec.trace.header.dtype
    target_names = list(spec_header_dtype.names)

    source_names = headers.dtype.names
    missing = [name for name in target_names if name not in source_names]
    if missing:
        msg = (
            f"SegySpec requires trace header fields not present in MDIO: {missing}. "
            f"Available MDIO header fields: {sorted(source_names)}."
        )
        raise ValueError(msg)

    # The export target must equal the dtype the headers were stored with at ingest, so route
    # through the same builder. That keeps ibm32 fields as float32, which flows into
    # SegyFactory.create_traces for IBM encoding instead of re-truncating to the raw uint32 slot.
    target_dtype = to_numpy_dtype(build_mdio_header_type(segy_spec))

    # Don't actually project if the dtype is already the same as the target dtype.
    if headers.dtype == target_dtype:
        return headers

    def _project_block(block: NDArray) -> NDArray:
        out = np.empty(block.shape, dtype=target_dtype)
        for name in target_names:
            out[name] = block[name]
        return out

    return headers.map_blocks(_project_block, dtype=target_dtype)


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
