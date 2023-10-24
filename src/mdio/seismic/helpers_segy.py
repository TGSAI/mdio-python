"""Helper functions for tinkering with SEG-Y related Zarr."""


from zarr import Group
from zarr import open_group
from zarr.errors import ContainsGroupError
from zarr.storage import FSStore

from mdio.core.exceptions import MDIOAlreadyExistsError


def create_zarr_hierarchy(store: FSStore, overwrite: bool) -> Group:
    """Create `zarr` hierarchy for SEG-Y files.

    Args:
        store: Output path where the converted output is written.
        overwrite: Toggle for overwriting existing store.

    Returns:
        Zarr Group instance for root of the file.

    Raises:
        MDIOAlreadyExistsError: If a file with data already exists.
    """
    root_group = open_group(store=store)

    try:
        root_group.create_group(name="data", overwrite=overwrite)
        root_group.create_group(name="metadata", overwrite=overwrite)
    except ContainsGroupError as e:
        msg = (
            f"An MDIO file with data already exists at {store.path}. "
            "If this is intentional, please specify 'overwrite=True'."
        )
        raise MDIOAlreadyExistsError(msg) from e

    return root_group
