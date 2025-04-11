"""Convenience utilities for writing to Zarr."""

from typing import TYPE_CHECKING
from typing import Any

from dask.array.core import normalize_chunks
from dask.array.rechunk import _balance_chunksizes


if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from zarr import Group


MAX_SIZE_LIVE_MASK = 512 * 1024**2


def write_attribute(name: str, attribute: Any, zarr_group: "Group") -> None:
    """Write a mappable to Zarr array or group attribute.

    Args:
        name: Name of the attribute.
        attribute: Mapping to write. Must be JSON serializable.
        zarr_group: Output group or array.
    """
    zarr_group.attrs[name] = attribute


def get_constrained_chunksize(
    shape: tuple[int, ...],
    dtype: "DTypeLike",
    max_bytes: int,
) -> tuple[int]:
    """Calculate the optimal chunk size for N-D array based on max_bytes.

    Args:
        shape: The shape of the array.
        dtype: The data dtype to be used in calculation.
        max_bytes: The maximum allowed number of bytes per chunk.

    Returns:
        A sequence of integers of calculated chunk sizes.
    """
    chunks = normalize_chunks("auto", shape, dtype=dtype, limit=max_bytes)
    return tuple(_balance_chunksizes(chunk)[0] for chunk in chunks)


def get_live_mask_chunksize(shape: tuple[int, ...]) -> tuple[int]:
    """Given a live_mask shape, calculate the optimal write chunk size.

    Args:
        shape: The shape of the array.

    Returns:
        A sequence of integers of calculated chunk sizes.
    """
    return get_constrained_chunksize(shape, "bool", MAX_SIZE_LIVE_MASK)
