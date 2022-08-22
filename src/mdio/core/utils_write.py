"""Convenience utilities for writing to Zarr."""


from typing import Any

import zarr


def write_attribute(name: str, attribute: Any, zarr_group: zarr.Group) -> None:
    """Write a mappable to Zarr array or group attribute.

    Args:
        name: Name of the attribute.
        attribute: Mapping to write. Must be JSON serializable.
        zarr_group: Output group or array.
    """
    zarr_group.attrs[name] = attribute
