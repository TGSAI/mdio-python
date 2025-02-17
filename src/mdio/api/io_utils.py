"""Utilities related to API functions and classes."""

from __future__ import annotations

import dask.array as da
import zarr


def process_url(
    url: str,
    disk_cache: bool,
) -> str:
    """Check read/write access to FSStore target and return FSStore with double caching.

    It can optionally use a file cache (`simplecache` protocol from fsspec) that
    is useful for remote stores. File cache is only useful for remote stores.

    The `storage_options` argument represents a set of parameters to be passed
    to the fsspec backend. Note that the format of `storage_options` is
    different if `disk_cache` is enabled or disabled, since `disk_cache`
    interanlly uses the simplecache protocol.

    Args:
        url: fsspec compliant url
        disk_cache: This enables fsspec's `simplecache` if True.

    Returns:
        Store with augmentations like cache, write verification etc.

    Examples:
        If we want to access an MDIO file from S3 without using disk caching,
        the simplecache protocol is not used, and therefore we only need to
        specify the s3 filesystem options:

        >>> from mdio.api.convenience import process_url
        >>>
        >>>
        >>> process_url(
        ...     url="s3://bucket/key",
        ...     mode="r",
        ...     storage_options={"key": "my_key", "secret": "my_secret"},
        ...     disk_cache=False,
        ... )

        On the other hand, if we want to use disk caching, we need to
        explicitly state that the options we are passing are for the S3
        filesystem:

        >>> process_url(
        ...     url="s3://bucket/key",
        ...     mode="r",
        ...     storage_options={"s3": {"key": "my_key", "secret": "my_secret"}},
        ...     disk_cache=True,
        ... )

        This allows us to pass options to the simplecache filesystem as well:

        >>> process_url(
        ...     url="s3://bucket/key",
        ...     mode="r",
        ...     storage_options={
        ...         "s3": {"key": "my_key", "secret": "my_secret"},
        ...         "simplecache": {"cache_storage": "custom/local/cache/path"},
        ...     },
        ...     disk_cache=True,
        ... )
    """
    if disk_cache is True:
        url = "::".join(["simplecache", url])

    return url


def open_zarr_array(group_handle: zarr.Group, name: str) -> zarr.Array:
    """Open Zarr array lazily using Zarr.

    Note: All other kwargs are ignored, used for API compatibility for dask backend.

    Args:
        group_handle: Group handle where the array is located
        name: Name of the array within the group

    Returns:
        Zarr array opened with default engine.
    """
    return group_handle[name]


def open_zarr_array_dask(group_handle: zarr.Group, name: str, **kwargs) -> da.Array:
    """Open Zarr array lazily using Dask.

    Note: All other kwargs get passed to dask.array.from_zarr()

    Args:
        group_handle: Group handle where the array is located
        name: Name of the array within the group
        **kwargs: Extra keyword arguments for Dask from_zarr.  # noqa: RST210

    Returns:
        Zarr array opened with Dask engine.
    """
    zarr_array = open_zarr_array(group_handle=group_handle, name=name)
    return da.from_array(zarr_array, **kwargs, inline_array=True)
