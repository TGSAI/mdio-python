"""Utilities related to API functions and classes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from dask.array import from_array

if TYPE_CHECKING:
    from dask.array import Array as DaskArray
    from zarr import Array as ZarrArray
    from zarr import Group as ZarrGroup


def process_url(url: str, disk_cache: bool) -> str:
    """Process URL based on options.

    File cache is only valid for remote stores. The LRU caching works on both remote and local.

    Args:
        url: FSSpec compliant url
        disk_cache: This enables FSSpec's `simplecache` if True.

    Returns:
        String to store with augmentations like cache, etc.

    Examples:
        If we want to access an MDIO file from S3 without using disk caching, the simplecache
        protocol is not used, and therefore we only need to specify the s3 filesystem options:

        >>> from mdio.api.convenience import process_url
        >>>
        >>>
        >>> process_url(
        ...     url="s3://bucket/key",
        ...     disk_cache=False,
        ... )
    """
    if disk_cache is True:
        url = f"simplecache::{url}"

    return url


def open_zarr_array(group_handle: ZarrGroup, name: str) -> ZarrArray:
    """Open Zarr array lazily using Zarr.

    Note: All other kwargs are ignored, used for API compatibility for dask backend.

    Args:
        group_handle: Group handle where the array is located
        name: Name of the array within the group

    Returns:
        Zarr array opened with default engine.
    """
    return group_handle[name]


def open_zarr_array_dask(group_handle: ZarrGroup, name: str, **kwargs: Any) -> DaskArray:  # noqa: ANN401
    """Open Zarr array lazily using Dask.

    Note: All other kwargs get passed to dask.array.from_zarr()

    Args:
        group_handle: Group handle where the array is located
        name: Name of the array within the group
        **kwargs: Extra keyword arguments for Dask from_zarr.

    Returns:
        Zarr array opened with Dask engine.
    """
    zarr_array = open_zarr_array(group_handle=group_handle, name=name)
    return from_array(zarr_array, **kwargs, inline_array=True)
