"""Utilities related to API functions and classes."""


from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import dask.array as da
import zarr


def process_url(
    url: str,
    mode: str,
    storage_options: dict[str, Any],
    memory_cache_size: int,
    disk_cache: bool,
) -> MutableMapping:
    """Check read/write access to FSStore target and return FSStore with double caching.

    It uses a file cache (simplecache protocol from FSSpec) and an in-memory
    Least Recently Used (LRU) cache implementation from zarr.

    File cache is only valid for remote stores. The LRU caching works
    on both remote and local.

    Args:
        url: FSSpec compliant url
        mode: Toggle for overwriting existing store
        storage_options: Storage options for the storage backend.
        memory_cache_size: Maximum in memory LRU cache size in bytes.
        disk_cache: This enables FSSpec's `simplecache` if True.

    Returns:
        Store with augmentations like cache, write verification etc.

    """
    # Append simplecache (disk caching) protocol
    # We need to change the storage options when caching is enabled.
    # Example below. This allows you to configure the cache protocol as well if needed.
    # storage_options_before = {'key': 'my_key', 'secret': 'my_secret'}
    # storage_options_after = {'s3:' {'key': 'my_key', 'secret': 'my_secret'},
    #                          'simplecache': {'cache_storage': '/my/cache/path'}}
    if disk_cache is True:
        url = "::".join(["simplecache", url])
        if "s3://" in url:
            storage_options = {"s3": storage_options}
        elif "gcs://" in url or "gs://" in url:
            storage_options = {"gcs": storage_options}

    # Flag for checking write access
    check = True if mode == "w" else False

    # TODO: Turning off write checking now because zarr has a bug.
    #  Get rid of this once bug is fixed.
    check = False

    # Let's open the FSStore and append LRU cache
    store = zarr.storage.FSStore(
        url=url,
        check=check,
        create=check,
        mode=mode,
        dimension_separator="/",
        **storage_options,
    )

    # Attach LRU Cache to store if requested.
    if memory_cache_size != 0:
        store = zarr.storage.LRUStoreCache(store=store, max_size=memory_cache_size)

    return store


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
    return da.from_array(zarr_array, **kwargs)
