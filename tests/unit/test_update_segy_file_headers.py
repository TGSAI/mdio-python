"""Tests for cheap in-place SEG-Y file header updates on MDIO stores."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
from segy.standards import get_segy_standard

from mdio import open_mdio
from mdio import to_mdio
from mdio import update_segy_file_headers
from mdio.exceptions import MDIONotFoundError
from mdio.segy.headers import BINARY_HEADER_ATTR
from mdio.segy.headers import SEGY_FILE_HEADER_VARIABLE
from mdio.segy.headers import TEXT_HEADER_ATTR
from mdio.segy.text_header import EXPECTED_COLS
from mdio.segy.text_header import EXPECTED_ROWS
from mdio.segy.text_header import validate_text_header

if TYPE_CHECKING:
    from pathlib import Path


def _well_formed_header(prefix: str = "C") -> str:
    """Build a 40x80 header with a distinctive prefix on each card."""
    rows = [f"{prefix}{i:02d}".ljust(EXPECTED_COLS) for i in range(1, EXPECTED_ROWS + 1)]
    return "\n".join(rows)


def _write_minimal_store(
    path: Path,
    *,
    with_header_var: bool = True,
    text_header: str | None = None,
    binary_header: dict[str, int] | None = None,
    samples: int = 8,
) -> Path:
    """Create a tiny MDIO store through the public write API."""
    dataset = xr.Dataset(
        {"amplitude": (("sample",), np.arange(samples, dtype=np.float32))},
        attrs={"attributes": {"defaultVariableName": "amplitude"}},
    )
    if with_header_var:
        dataset[SEGY_FILE_HEADER_VARIABLE] = ((), "")
        attrs: dict[str, object] = {}
        if text_header is not None:
            attrs[TEXT_HEADER_ATTR] = text_header
        if binary_header is not None:
            attrs[BINARY_HEADER_ATTR] = binary_header
        if attrs:
            dataset[SEGY_FILE_HEADER_VARIABLE].attrs.update(attrs)
    to_mdio(dataset, path, mode="w")
    return path


def _stored_headers(path: Path) -> tuple[str, dict[str, int]]:
    """Read header attrs back through ``open_mdio``."""
    attrs = open_mdio(path)[SEGY_FILE_HEADER_VARIABLE].attrs
    return attrs[TEXT_HEADER_ATTR], dict(attrs[BINARY_HEADER_ATTR])


class TestUpdateSegyFileHeaders:
    """In-place header updates write attrs only and fill missing fields."""

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        """Unknown store path raises ``MDIONotFoundError``."""
        with pytest.raises(MDIONotFoundError, match="not found"):
            update_segy_file_headers(tmp_path / "missing.mdio")

    def test_updates_text_header_only(self, tmp_path: Path) -> None:
        """Provided text replaces the stored text; binary stays put."""
        original_binary = {"job_id": 7, "sample_interval": 2000, "samples_per_trace": 8}
        store = _write_minimal_store(
            tmp_path / "headers.mdio",
            text_header=_well_formed_header("A"),
            binary_header=original_binary,
        )
        new_text = _well_formed_header("B")

        result = update_segy_file_headers(store, text_header=new_text)

        stored_text, stored_binary = _stored_headers(store)
        assert result.text_header == new_text
        assert stored_text == new_text
        assert stored_binary == original_binary

    def test_merges_binary_header_fields(self, tmp_path: Path) -> None:
        """User binary fields overlay the existing mapping."""
        store = _write_minimal_store(
            tmp_path / "headers.mdio",
            text_header=_well_formed_header(),
            binary_header={"job_id": 1, "line_num": 2, "sample_interval": 2000},
        )

        result = update_segy_file_headers(store, binary_header={"job_id": 99})

        _, stored_binary = _stored_headers(store)
        assert result.binary_header["job_id"] == 99
        assert stored_binary["job_id"] == 99
        assert stored_binary["line_num"] == 2
        assert stored_binary["sample_interval"] == 2000

    def test_sanitizes_short_text_header(self, tmp_path: Path) -> None:
        """Short user text is padded to the 40x80 card layout."""
        store = _write_minimal_store(
            tmp_path / "headers.mdio",
            text_header=_well_formed_header(),
            binary_header={"job_id": 1},
        )

        result = update_segy_file_headers(store, text_header="C01 CLIENT")

        validate_text_header(result.text_header)
        assert result.text_header.split("\n")[0].startswith("C01 CLIENT")

    def test_creates_missing_variable_with_defaults(self, tmp_path: Path) -> None:
        """Store without ``segy_file_header`` gets a scalar var plus Rev1 defaults."""
        store = _write_minimal_store(tmp_path / "no_headers.mdio", with_header_var=False, samples=8)

        result = update_segy_file_headers(store)

        validate_text_header(result.text_header)
        stored_text, stored_binary = _stored_headers(store)
        assert stored_text == result.text_header
        assert stored_binary["samples_per_trace"] == 8
        assert stored_binary["segy_revision_major"] == 1
        assert stored_binary["segy_revision_minor"] == 0
        assert "sample_interval" in stored_binary

    def test_fills_missing_attrs_on_existing_variable(self, tmp_path: Path) -> None:
        """Empty header variable receives default text and binary attrs."""
        store = _write_minimal_store(tmp_path / "empty_attrs.mdio", with_header_var=True)

        result = update_segy_file_headers(store)

        validate_text_header(result.text_header)
        _, stored_binary = _stored_headers(store)
        assert stored_binary["samples_per_trace"] == 8

    def test_user_binary_overlays_defaults_when_missing(self, tmp_path: Path) -> None:
        """Partial user binary sits on top of the generated default header."""
        store = _write_minimal_store(tmp_path / "partial.mdio", with_header_var=False)

        result = update_segy_file_headers(store, binary_header={"job_id": 42})

        assert result.binary_header["job_id"] == 42
        assert result.binary_header["samples_per_trace"] == 8
        assert result.binary_header["segy_revision_major"] == 1

    def test_template_controls_default_revision(self, tmp_path: Path) -> None:
        """Optional SegySpec template drives default binary field set and revision."""
        store = _write_minimal_store(tmp_path / "rev2.mdio", with_header_var=False)

        result = update_segy_file_headers(store, template=get_segy_standard(2.0))

        assert result.binary_header["segy_revision_major"] == 2
        assert result.binary_header["segy_revision_minor"] == 0
        assert "extended_samples_per_trace" in result.binary_header

    def test_does_not_rewrite_amplitude_payload(self, tmp_path: Path) -> None:
        """Header update leaves the data array bytes unchanged."""
        store = _write_minimal_store(
            tmp_path / "payload.mdio",
            text_header=_well_formed_header(),
            binary_header={"job_id": 1},
        )
        before = open_mdio(store)["amplitude"].values.copy()

        update_segy_file_headers(store, binary_header={"job_id": 2})

        after = open_mdio(store)["amplitude"].values
        np.testing.assert_array_equal(before, after)

    def test_rejects_non_integer_binary_value(self, tmp_path: Path) -> None:
        """Non-integer binary field values raise ``ValueError``."""
        store = _write_minimal_store(
            tmp_path / "bad.mdio",
            text_header=_well_formed_header(),
            binary_header={"job_id": 1},
        )
        with pytest.raises(ValueError, match="must be an integer"):
            update_segy_file_headers(store, binary_header={"job_id": "not-a-number"})  # type: ignore[dict-item]

    def test_normalizes_encoded_revision(self, tmp_path: Path) -> None:
        """Encoded ``segy_revision`` is stored as major/minor like ingest."""
        store = _write_minimal_store(
            tmp_path / "rev.mdio",
            text_header=_well_formed_header(),
            binary_header={"job_id": 1},
        )

        result = update_segy_file_headers(store, binary_header={"segy_revision": 256})

        assert "segy_revision" not in result.binary_header
        assert result.binary_header["segy_revision_major"] == 1
        assert result.binary_header["segy_revision_minor"] == 0

    def test_unknown_binary_key_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Fields outside the SEG-Y template warn; export can break if they are added."""
        store = _write_minimal_store(
            tmp_path / "unknown.mdio",
            text_header=_well_formed_header(),
            binary_header={"job_id": 1},
        )
        with caplog.at_level(logging.WARNING, logger="mdio.segy.headers"):
            update_segy_file_headers(store, binary_header={"not_a_segy_field": 1})
        assert any("not_a_segy_field" in record.message for record in caplog.records)
