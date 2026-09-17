"""Tests for cheap in-place CRS updates on MDIO stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
import zarr

from mdio import open_mdio
from mdio import to_mdio
from mdio import update_crs
from mdio.exceptions import MDIONotFoundError
from tests.unit.testing_helpers import zarr_attrs_tree

if TYPE_CHECKING:
    from pathlib import Path


def _write_minimal_store(path: Path, crs: str | None = None) -> Path:
    """Create a tiny MDIO store through the public write API, optionally seeding a CRS."""
    dataset = xr.Dataset(
        {"amplitude": (("sample",), np.arange(8, dtype=np.float32))},
        coords={"sample": np.arange(8, dtype=np.int64) * 4},
        attrs={"attributes": {"defaultVariableName": "amplitude"}},
    )
    to_mdio(dataset, path, mode="w")
    if crs is not None:
        update_crs(path, crs)
    return path


class TestUpdateCrs:
    """In-place CRS updates write root attrs only."""

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        """Unknown store path raises ``MDIONotFoundError``."""
        with pytest.raises(MDIONotFoundError, match="not found"):
            update_crs(tmp_path / "missing.mdio", "EPSG:32610")

    def test_sets_crs_on_store_without_one(self, tmp_path: Path) -> None:
        """A store with no CRS receives the new root attribute."""
        store = _write_minimal_store(tmp_path / "crs.mdio")

        update_crs(store, "EPSG:32610")

        assert open_mdio(store).attrs["crs"] == "EPSG:32610"

    def test_replaces_existing_crs(self, tmp_path: Path) -> None:
        """A later update overwrites the stored CRS string."""
        store = _write_minimal_store(tmp_path / "crs.mdio", crs="EPSG:4326")

        update_crs(store, "EPSG:32610")

        assert open_mdio(store).attrs["crs"] == "EPSG:32610"

    def test_removes_crs(self, tmp_path: Path) -> None:
        """``None`` deletes the root CRS attribute."""
        store = _write_minimal_store(tmp_path / "crs.mdio", crs="EPSG:32610")

        update_crs(store, None)

        assert "crs" not in open_mdio(store).attrs

    def test_remove_when_absent_is_noop(self, tmp_path: Path) -> None:
        """Removing a missing CRS does not fail."""
        store = _write_minimal_store(tmp_path / "crs.mdio")

        update_crs(store, None)

        assert "crs" not in open_mdio(store).attrs

    def test_does_not_rewrite_amplitude_payload(self, tmp_path: Path) -> None:
        """CRS update leaves the data array bytes unchanged."""
        store = _write_minimal_store(tmp_path / "payload.mdio")
        before = open_mdio(store)["amplitude"].values.copy()

        update_crs(store, "EPSG:32610")

        after = open_mdio(store)["amplitude"].values
        np.testing.assert_array_equal(before, after)

    def test_preserves_sibling_attributes(self, tmp_path: Path) -> None:
        """Root ``crs`` is the only attribute key that changes."""
        dataset = xr.Dataset(
            {"amplitude": (("sample",), np.arange(8, dtype=np.float32))},
            coords={"sample": np.arange(8, dtype=np.int64) * 4},
            attrs={"attributes": {"defaultVariableName": "amplitude"}, "name": "survey"},
        )
        dataset["amplitude"].attrs["statsV1"] = '{"count": 8}'
        dataset["amplitude"].attrs["long_name"] = "amplitude"
        store = tmp_path / "attrs.mdio"
        to_mdio(dataset, store, mode="w")
        before = zarr_attrs_tree(store)

        update_crs(store, "EPSG:32610")

        after = zarr_attrs_tree(store)
        assert after[""]["crs"] == "EPSG:32610"
        assert {key: value for key, value in after[""].items() if key != "crs"} == before[""]
        assert {key: attrs for key, attrs in after.items() if key != ""} == {
            key: attrs for key, attrs in before.items() if key != ""
        }

    def test_preserves_unconsolidated_root_attributes_in_v2(self, tmp_path: Path) -> None:
        """A stale v2 consolidated snapshot cannot erase newer root attributes."""
        with zarr.config.set({"default_zarr_format": 2}):
            store = _write_minimal_store(tmp_path / "stale-attrs.mdio")
            root = zarr.open_group(store.as_posix(), mode="r+", use_consolidated=False)
            root.attrs["addedAfterConsolidation"] = {"keep": True}

            update_crs(store, "EPSG:32610")

            attrs = open_mdio(store).attrs
            assert attrs["addedAfterConsolidation"] == {"keep": True}
            assert attrs["crs"] == "EPSG:32610"

    def test_consolidates_v2_store_when_default_format_is_v3(self, tmp_path: Path) -> None:
        """V2 consolidation follows the opened store, not the process default format."""
        with zarr.config.set({"default_zarr_format": 2}):
            store = _write_minimal_store(tmp_path / "format-mismatch.mdio")
        with zarr.config.set({"default_zarr_format": 3}):
            update_crs(store, "EPSG:32610")

        root = zarr.open_group(store.as_posix(), mode="r", use_consolidated=True)
        assert root.metadata.zarr_format == 2
        assert root.attrs["crs"] == "EPSG:32610"
