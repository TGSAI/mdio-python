"""Tests for generic dataset metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace

from mdio.ingestion.metadata import _add_grid_override_to_metadata


def _make_dataset(attributes: dict | None) -> SimpleNamespace:
    """Build a minimal stand-in for Dataset with a nested ``metadata.attributes``."""
    return SimpleNamespace(metadata=SimpleNamespace(attributes=attributes))


class TestAddGridOverrideToMetadata:
    """Tests for ``_add_grid_override_to_metadata``."""

    def test_initializes_attributes_dict_when_none(self) -> None:
        """A ``None`` attributes dict gets replaced with an empty dict before insertion."""
        dataset = _make_dataset(attributes=None)
        _add_grid_override_to_metadata(dataset, grid_overrides=None)
        assert dataset.metadata.attributes == {}

    def test_adds_grid_overrides_when_provided(self) -> None:
        """Grid overrides should land under the ``gridOverrides`` key."""
        dataset = _make_dataset(attributes=None)
        overrides = {"HasDuplicates": True, "chunksize": 4}
        _add_grid_override_to_metadata(dataset, grid_overrides=overrides)
        assert dataset.metadata.attributes == {"gridOverrides": overrides}

    def test_preserves_existing_attributes(self) -> None:
        """Existing attribute keys should be preserved when adding overrides."""
        dataset = _make_dataset(attributes={"existing": "value"})
        overrides = {"NonBinned": True}
        _add_grid_override_to_metadata(dataset, grid_overrides=overrides)
        assert dataset.metadata.attributes == {"existing": "value", "gridOverrides": overrides}

    def test_no_overrides_leaves_attributes_untouched(self) -> None:
        """Passing ``None`` overrides must not introduce a ``gridOverrides`` key."""
        dataset = _make_dataset(attributes={"existing": "value"})
        _add_grid_override_to_metadata(dataset, grid_overrides=None)
        assert dataset.metadata.attributes == {"existing": "value"}
