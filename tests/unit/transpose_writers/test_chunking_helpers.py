"""Unit tests for chunking module helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
import zarr

from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import Blosc
from mdio.constants import ZarrFormat
from mdio.transpose_writers.chunking import _normalize_chunk_grid
from mdio.transpose_writers.chunking import _normalize_compressor
from mdio.transpose_writers.chunking import _normalize_new_variable
from mdio.transpose_writers.chunking import _remove_fillvalue_attrs
from mdio.transpose_writers.chunking import _validate_inputs

if TYPE_CHECKING:
    from pathlib import Path


class TestRemoveFillvalueAttrs:
    """Tests for _remove_fillvalue_attrs helper function."""

    @pytest.mark.parametrize("zarr_format", [ZarrFormat.V2, ZarrFormat.V3])
    def test_remove_fillvalue_after_mdio_serialization(self, tmp_path: Path, zarr_format: ZarrFormat) -> None:
        """Test that _FillValue is removed after MDIO serialization in both Zarr v2 and v3."""
        # Create dataset with NaN values (will add _FillValue on serialization)
        data = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 4.0]], dtype=np.float32)
        ds = xr.Dataset(
            {"var1": (["x", "y"], data, {"units": "meters"})},
            coords={"x": (["x"], [0, 1], {"axis": "X"})},
        )

        # Write and read back with MDIO
        mdio_path = tmp_path / f"test_{zarr_format}.mdio"
        with zarr.config.set(default_zarr_format=zarr_format):
            to_mdio(ds, mdio_path, mode="w")
            ds_read = open_mdio(mdio_path)

            # Apply function and verify _FillValue is removed everywhere
            _remove_fillvalue_attrs(ds_read)

            for var_name in list(ds_read.data_vars) + list(ds_read.coords):
                assert "_FillValue" not in ds_read[var_name].attrs

            # Verify other attributes preserved
            assert ds_read["var1"].attrs["units"] == "meters"
            assert ds_read["x"].attrs["axis"] == "X"


class TestValidateInputs:
    """Tests for _validate_inputs helper function."""

    @pytest.mark.parametrize(
        ("new_variable", "chunk_grid", "compressor", "should_pass"),
        [
            ("var1", "grid", "comp", True),
            (["var1", "var2"], "grid", "comp", True),
            ("var1", "grid", None, True),
            (123, "grid", "comp", False),
            ([], "grid", "comp", False),
            (["var1", 123], "grid", "comp", False),  # non-string in list
            ("var1", [], "comp", False),
            ("var1", "grid", [], False),
        ],
    )
    def test_validation(
        self,
        new_variable: str | list | int,
        chunk_grid: str | list,
        compressor: str | list | None,
        should_pass: bool,
    ) -> None:
        """Test input validation with various combinations."""
        grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=(10, 10)))
        comp = Blosc(cname="zstd", clevel=5, shuffle="shuffle")

        # Replace placeholders
        if chunk_grid == "grid":
            chunk_grid = grid
        if compressor == "comp":
            compressor = comp

        if should_pass:
            _validate_inputs(new_variable, chunk_grid, compressor)  # type: ignore[arg-type]
        else:
            with pytest.raises((TypeError, ValueError)):
                _validate_inputs(new_variable, chunk_grid, compressor)  # type: ignore[arg-type]


class TestNormalizeNewVariable:
    """Tests for _normalize_new_variable helper function."""

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            ("var1", ["var1"]),
            (["var1", "var2"], ["var1", "var2"]),
        ],
    )
    def test_normalize(self, input_value: str | list[str], expected: list[str]) -> None:
        """Test new_variable normalization."""
        result = _normalize_new_variable(input_value)
        assert result == expected


class TestNormalizeChunkGrid:
    """Tests for _normalize_chunk_grid helper function."""

    def test_broadcast_and_match(self) -> None:
        """Test chunk grid broadcasting and matching."""
        grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=(10, 10, 10)))
        grids = [grid] * 3

        # Single grid broadcasts
        assert len(_normalize_chunk_grid(grid, 3)) == 3

        # Single-element list broadcasts
        result = _normalize_chunk_grid([grid], 3)
        assert len(result) == 3
        assert result == grids

        # List matches length
        result = _normalize_chunk_grid(grids, 3)
        assert len(result) == 3
        assert result == grids

        # Mismatch raises error
        with pytest.raises(
            ValueError, match="chunk_grid list length must be 1 or equal to the number of new variables"
        ):
            _normalize_chunk_grid([grid, grid], 3)


class TestNormalizeCompressor:
    """Tests for _normalize_compressor helper function."""

    def test_broadcast_and_match(self) -> None:
        """Test compressor broadcasting and matching."""
        comp = Blosc(cname="zstd", clevel=5, shuffle="shuffle")

        # None broadcasts
        assert all(c is None for c in _normalize_compressor(None, 3))

        # Single compressor broadcasts
        assert len(_normalize_compressor(comp, 3)) == 3

        # Single-element list broadcasts
        result = _normalize_compressor([comp], 3)
        assert len(result) == 3
        assert all(c == comp for c in result)

        # List with None entries
        result = _normalize_compressor([comp, None, comp], 3)
        assert result[1] is None

        # Mismatch raises error
        with pytest.raises(
            ValueError, match="compressor list length must be 1 or equal to the number of new variables"
        ):
            _normalize_compressor([comp, comp], 3)
