"""Generic dataset metadata helpers for ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from mdio.builder.schemas import Dataset


def _add_grid_override_to_metadata(dataset: Dataset, grid_overrides: dict[str, Any] | None) -> None:
    """Add grid override to Dataset metadata if needed."""
    if dataset.metadata.attributes is None:
        dataset.metadata.attributes = {}

    if grid_overrides is not None:
        dataset.metadata.attributes["gridOverrides"] = grid_overrides
