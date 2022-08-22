"""Convenience utilities for writing to Zarr."""


from typing import Any

import zarr


def write_attribute(name: str, attribute: Any, zarr_group: zarr.Group) -> None:
    """Write text headers to a file.

    Parameters
    ----------
    name : str
        Name of the attribute
    attribute : Any
        Attribute to write into `zarr.Group`. Must be JSON serializable.
    zarr_group : zarr.Group
        Path or buffer for the output.

    Returns
    -------
    None

    """
    zarr_group.attrs[name] = attribute
