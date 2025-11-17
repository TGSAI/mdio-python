"""Sharding operations for MDIO datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import Variable


def from_variable(
    variable: Variable,
    num_shards: int,
    *,
    shard_dimension: str | None = None,
) -> list[Variable]:
    """Shard a Variable across multiple pieces for distributed processing.

    Args:
        variable: The input Variable to shard.
        num_shards: Number of shards to create.
        shard_dimension: Dimension along which to shard. If None,
            automatically selects the largest dimension.

    Returns:
        List of Variable shards.

    Raises:
        NotImplementedError: If sharding operations are not yet implemented.
    """
    msg = "Sharding operations are not yet implemented"
    raise NotImplementedError(msg)
