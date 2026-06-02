"""Tests for generic dataset metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace

from mdio.ingestion.metadata import add_grid_override_to_metadata
from mdio.segy.geometry import GridOverrides


def _make_dataset(attributes: dict | None) -> SimpleNamespace:
    """Build a minimal stand-in for Dataset with a nested ``metadata.attributes``."""
    return SimpleNamespace(metadata=SimpleNamespace(attributes=attributes))


class TestAddGridOverrideToMetadata:
    """Tests for ``add_grid_override_to_metadata``."""

    def test_initializes_attributes_dict_when_none(self) -> None:
        """A ``None`` attributes dict gets replaced with an empty dict before insertion."""
        dataset = _make_dataset(attributes=None)
        add_grid_override_to_metadata(dataset, grid_overrides=None)
        assert dataset.metadata.attributes == {}

    def test_adds_grid_overrides_when_provided(self) -> None:
        """Active grid overrides should serialize under the ``gridOverrides`` key."""
        dataset = _make_dataset(attributes=None)
        overrides = GridOverrides(has_duplicates=True, chunksize=4)
        add_grid_override_to_metadata(dataset, grid_overrides=overrides)
        assert dataset.metadata.attributes == {
            "gridOverrides": {"HasDuplicates": True, "chunksize": 4},
        }

    def test_preserves_existing_attributes(self) -> None:
        """Existing attribute keys should be preserved when adding overrides."""
        dataset = _make_dataset(attributes={"existing": "value"})
        overrides = GridOverrides(non_binned=True)
        add_grid_override_to_metadata(dataset, grid_overrides=overrides)
        assert dataset.metadata.attributes == {
            "existing": "value",
            "gridOverrides": {"NonBinned": True},
        }

    def test_no_overrides_leaves_attributes_untouched(self) -> None:
        """Passing ``None`` overrides must not introduce a ``gridOverrides`` key."""
        dataset = _make_dataset(attributes={"existing": "value"})
        add_grid_override_to_metadata(dataset, grid_overrides=None)
        assert dataset.metadata.attributes == {"existing": "value"}
