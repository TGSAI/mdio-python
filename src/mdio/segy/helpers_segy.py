"""Helper functions for tinkering with SEG-Y related Zarr."""

from typing import TYPE_CHECKING

from zarr.errors import ContainsGroupError

from mdio.exceptions import MDIOAlreadyExistsError

if TYPE_CHECKING:
    from zarr import Group


def create_zarr_hierarchy(root_group: "Group", overwrite: bool) -> "Group":
    """Create `zarr` hierarchy for SEG-Y files.

    Args:
        root_group: Output root group where data will be written.
        overwrite: Toggle for overwriting existing store.

    Returns:
        Zarr Group instance for root of the file.

    Raises:
        MDIOAlreadyExistsError: If a file with data already exists.
    """
    try:
        root_group.create_group(name="data", overwrite=overwrite)
        root_group.create_group(name="metadata", overwrite=overwrite)
    except ContainsGroupError as e:
        msg = (
            f"An MDIO file with data already exists at {root_group.store_path}. "
            "If this is intentional, please specify 'overwrite=True'."
        )
        raise MDIOAlreadyExistsError(msg) from e

    return root_group
