"""Tests for attaching SEG-Y text/binary file headers to xarray datasets."""

from __future__ import annotations

import base64
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from xarray import DataArray as xr_DataArray
from xarray import Dataset as xr_Dataset

from mdio.ingestion.segy.file_headers import _add_segy_file_headers


def _valid_text_header() -> str:
    """Build a SEG-Y text header with 40 rows of exactly 80 chars."""
    return "\n".join(["X" * 80] * 40)


def _make_segy_info(
    text_header: str | None = None,
    binary_header_dict: dict | None = None,
    raw_binary_headers: bytes = b"\x00" * 400,
) -> SimpleNamespace:
    return SimpleNamespace(
        text_header=text_header if text_header is not None else _valid_text_header(),
        binary_header_dict=binary_header_dict if binary_header_dict is not None else {"sample_interval": 4000},
        raw_binary_headers=raw_binary_headers,
    )


def _empty_dataset() -> xr_Dataset:
    return xr_Dataset({"amplitude": xr_DataArray(np.zeros(2, dtype=np.float32), dims=["inline"])})


class TestAddSegyFileHeaders:
    """Tests for ``_add_segy_file_headers``."""

    def test_disabled_returns_dataset_unchanged(self) -> None:
        """When the save flag is off the dataset must not be modified."""
        info = _make_segy_info()
        ds = _empty_dataset()
        with patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "false"}):
            result = _add_segy_file_headers(ds, info)

        assert "segy_file_header" not in result

    def test_attaches_headers_when_enabled(self) -> None:
        """Enabling the flag should attach text + binary header attrs."""
        info = _make_segy_info()
        ds = _empty_dataset()
        with patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "true"}):
            result = _add_segy_file_headers(ds, info)

        attrs = result["segy_file_header"].attrs
        assert attrs["textHeader"] == info.text_header
        assert attrs["binaryHeader"] == info.binary_header_dict
        assert "rawBinaryHeader" not in attrs

    def test_attaches_raw_binary_when_raw_flag_enabled(self) -> None:
        """``raw_headers`` should add the base64-encoded raw binary headers."""
        info = _make_segy_info(raw_binary_headers=b"abc")
        ds = _empty_dataset()
        env = {
            "MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "true",
            "MDIO__IMPORT__RAW_HEADERS": "true",
        }
        with patch.dict(os.environ, env):
            result = _add_segy_file_headers(ds, info)

        encoded = base64.b64encode(b"abc").decode("ascii")
        assert result["segy_file_header"].attrs["rawBinaryHeader"] == encoded

    def test_invalid_row_count_raises(self) -> None:
        """Text header without 40 rows must raise."""
        bad_text = "\n".join(["X" * 80] * 39)
        info = _make_segy_info(text_header=bad_text)
        ds = _empty_dataset()
        with (
            patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "true"}),
            pytest.raises(ValueError, match="Invalid text header line count"),
        ):
            _add_segy_file_headers(ds, info)

    def test_invalid_column_count_raises(self) -> None:
        """Text header rows shorter than 80 chars must raise."""
        bad_rows = ["X" * 80] * 40
        bad_rows[5] = "X" * 79
        info = _make_segy_info(text_header="\n".join(bad_rows))
        ds = _empty_dataset()
        with (
            patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "true"}),
            pytest.raises(ValueError, match="Invalid text header line widths"),
        ):
            _add_segy_file_headers(ds, info)
