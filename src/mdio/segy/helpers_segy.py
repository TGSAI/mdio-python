"""Helper functions for tinkering with SEG-Y related Zarr."""


from math import prod

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
            f"An MDIO file with data already exists at '{store.path}'. "
            "If this is intentional, please specify 'overwrite=True'."
        )
        raise MDIOAlreadyExistsError(msg) from e

    return root_group


# TODO: This is not used right now, but it is a template for what we can do for
#  automatic chunk size determination based on shape of the arrays etc.
def infer_header_chunksize(orig_chunks, orig_shape, target_size=2**26, length=240):
    """Infer larger chunks based on target chunk filesize.

    This tool takes an original chunking scheme, the full shape of the
    original array, a target size (in bytes) and length of the `array` or
    `struct` to calculate a multidimensional scalar for smaller arrays.

    Use case is: Seismic data has 1 extra time/depth dimension, which doesn't
    exist in headers or spatial live mask. So we can make chunk size bigger
    for these flatter arrays.

    This module infers a scalar based on the parameters and returns a new
    chunking scheme.

    Args:
        orig_chunks: Original array chunks.
        orig_shape: Original array shape.
        target_size: Uncompressed, expected size of each chunk. This is much
            larger than the ideal 1MB because on metadata, after compression,
            the size goes down by 10x. Default: 32 MB.
        length: Length of the multidimensional array's dtype.
            Default is 240-bytes.

    Returns:
            Tuple of adjusted chunk sizes.
    """
    orig_bytes = prod(orig_chunks) * length

    # Size scalar in bytes
    scalar = target_size / orig_bytes

    # Divide than into chunks (root of the scalar based on length of dims)
    # Then round it to the nearest integer.
    scalar = round(scalar ** (1 / len(orig_chunks)))

    # Scale chunks by inferred isotropic scalar.
    new_chunks = [dim_chunk * scalar for dim_chunk in orig_chunks]

    # Set it to max if after scaling, it is larger than the max values.
    new_chunks = [
        min(dim_new, dim_orig)
        for dim_new, dim_orig in zip(new_chunks, orig_shape)  # noqa: B905
    ]

    # Special case if the new_chunks are larger than 80% the original shape.
    # In this case we want one chunk.
    if prod(new_chunks) > 0.8 * prod(orig_shape):
        new_chunks = orig_shape

    return new_chunks
