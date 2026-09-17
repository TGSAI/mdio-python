"""Shared helpers for unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import zarr

if TYPE_CHECKING:
    from pathlib import Path


def zarr_attrs_tree(path: Path) -> dict[str, dict[str, object]]:
    """Return user attrs for the root group and every descendant node."""
    root = zarr.open_group(path.as_posix(), mode="r")
    tree: dict[str, dict[str, object]] = {"": dict(root.attrs)}
    for name, node in root.members(max_depth=None):
        tree[name] = dict(node.attrs)
    return tree
