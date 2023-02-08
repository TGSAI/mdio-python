"""Convenience APIs for working with MDIO files."""


from __future__ import annotations

import zarr

from mdio.api.io_utils import process_url


def copy_mdio(
    source,
    dest_path_or_buffer: str,
    excludes="",
    includes="",
    storage_options: dict | None = None,
    overwrite: bool = False,
) -> None:
    """Copy MDIO file.

    Can also copy with empty data to be filled later. See `excludes`
    and `includes` parameters.

    More documentation about `excludes` and `includes` can be found
    in Zarr's documentation in `zarr.convenience.copy_store`.

    Args:
        source: MDIO reader or accessor instance. Data will be copied from here
        dest_path_or_buffer: Destination path. Could be any FSSpec mapping.
        excludes: Data to exclude during copy. i.e. `chunked_012`. The raw data
            won't be copied, but it will create an empty array to be filled.
            If left blank, it will copy everything.
        includes: Data to include during copy. i.e. `trace_headers`. If this is
            not specified, and certain data is excluded, it will not copy headers.
            If you want to preserve headers, specify `trace_headers`. If left blank,
            it will copy everything except specified in `excludes` parameter.
        storage_options: Storage options for the cloud storage backend.
            Default is None (will assume anonymous).
        overwrite: Overwrite destination or not.

    """
    if storage_options is None:
        storage_options = {}

    dest_store = process_url(
        url=dest_path_or_buffer,
        mode="w",
        storage_options=storage_options,
        memory_cache_size=0,
        disk_cache=False,
    )

    if_exists = "replace" if overwrite is True else "raise"

    zarr.copy_store(
        source=source.store,
        dest=dest_store,
        excludes=excludes,
        includes=includes,
        if_exists=if_exists,
    )

    if len(excludes) > 0:
        data_path = "/".join(["data", excludes])
        source_array = source.root[data_path]
        dimension_separator = source_array._dimension_separator

        zarr.empty_like(
            source_array,
            store=dest_store,
            path=data_path,
            overwrite=overwrite,
            dimension_separator=dimension_separator,
        )
