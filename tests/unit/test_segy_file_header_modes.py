"""Tests for ``add_segy_file_headers`` mode handling.

Covers the three values of ``MDIO__IMPORT__SAVE_SEGY_FILE_HEADER``:
0 skips, 1 raises on a malformed text header, 2 corrects a malformed text header.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest
import xarray as xr

from mdio.ingestion.segy.file_headers import add_segy_file_headers
from mdio.segy.file import SegyFileInfo


def _well_formed_header() -> str:
    """Build a 40x80 header where each row reads ``Cnn ...spaces``."""
    return "\n".join([f"C{i:02d}".ljust(80) for i in range(1, 41)])


def _malformed_header() -> str:
    """Header with a NUL byte injected into row 0; valid 80x40 layout otherwise."""
    rows = [f"C{i:02d}".ljust(80) for i in range(1, 41)]
    rows[0] = "C01\x00" + " " * 76
    return "\n".join(rows)


def _replacement_char_header() -> str:
    """Build a 40x80 header with U+FFFD scattered through the last three cards."""
    rows = [f"C{i:02d}".ljust(80) for i in range(1, 41)]
    rows[37] = "\ufffdC38" + " " * 76
    rows[38] = "\ufffdC39" + " " * 76
    rows[39] = "\ufffdC40 END EBCDIC" + " " * 65
    return "\n".join(rows)


def _segy_info(text_header: str) -> SegyFileInfo:
    """Minimal SegyFileInfo fixture with the given text header."""
    return SegyFileInfo(
        num_traces=1,
        sample_labels=None,
        text_header=text_header,
        binary_header_dict={"job_id": 1},
        raw_binary_headers=b"",
        coordinate_scalar=1,
    )


class TestSaveSegyFileHeaderModes:
    """Mode 0 skips, mode 1 strict, mode 2 lenient."""

    def test_mode_zero_skips_header_save(self) -> None:
        """Mode 0 leaves the dataset without a ``segy_file_header`` variable."""
        ds = xr.Dataset()
        with patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "0"}):
            result = add_segy_file_headers(ds, _segy_info(_malformed_header()))

        assert "segy_file_header" not in result

    def test_mode_one_accepts_well_formed(self) -> None:
        """Mode 1 stores a well-formed header verbatim."""
        ds = xr.Dataset()
        header = _well_formed_header()
        with patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "1"}):
            result = add_segy_file_headers(ds, _segy_info(header))

        assert result["segy_file_header"].attrs["textHeader"] == header

    def test_mode_one_raises_on_malformed(self) -> None:
        """Mode 1 raises on a NUL byte in the header."""
        ds = xr.Dataset()
        with (
            patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "1"}),
            pytest.raises(ValueError, match="non-ASCII or non-printable"),
        ):
            add_segy_file_headers(ds, _segy_info(_malformed_header()))

    def test_mode_one_raises_on_replacement_char(self) -> None:
        """Mode 1 raises on U+FFFD."""
        ds = xr.Dataset()
        with (
            patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "1"}),
            pytest.raises(ValueError, match="non-ASCII or non-printable"),
        ):
            add_segy_file_headers(ds, _segy_info(_replacement_char_header()))

    def test_mode_two_corrects_malformed(self, caplog: pytest.LogCaptureFixture) -> None:
        """Mode 2 repairs a NUL byte and stores a 40x80 header."""
        ds = xr.Dataset()
        with (
            patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "2"}),
            caplog.at_level(logging.WARNING, logger="mdio.ingestion.segy.file_headers"),
        ):
            result = add_segy_file_headers(ds, _segy_info(_malformed_header()))

        stored = result["segy_file_header"].attrs["textHeader"]
        assert "\x00" not in stored
        assert all(len(row) == 80 for row in stored.split("\n"))
        assert len(stored.split("\n")) == 40
        assert any("Correcting malformed" in record.message for record in caplog.records)

    def test_mode_two_corrects_replacement_char(self, caplog: pytest.LogCaptureFixture) -> None:
        """Mode 2 repairs U+FFFD and stores ASCII-encodable bytes."""
        ds = xr.Dataset()
        with (
            patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "2"}),
            caplog.at_level(logging.WARNING, logger="mdio.ingestion.segy.file_headers"),
        ):
            result = add_segy_file_headers(ds, _segy_info(_replacement_char_header()))

        stored = result["segy_file_header"].attrs["textHeader"]
        assert "\ufffd" not in stored
        stored.replace("\n", "").encode("ascii")
        assert any("Correcting malformed" in record.message for record in caplog.records)

    def test_mode_two_passes_through_well_formed(self, caplog: pytest.LogCaptureFixture) -> None:
        """Mode 2 stays silent and bit-identical on well-formed input."""
        ds = xr.Dataset()
        header = _well_formed_header()
        with (
            patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "2"}),
            caplog.at_level(logging.WARNING, logger="mdio.ingestion.segy.file_headers"),
        ):
            result = add_segy_file_headers(ds, _segy_info(header))

        assert result["segy_file_header"].attrs["textHeader"] == header
        assert not any("Correcting" in record.message for record in caplog.records)

    def test_mode_two_repairs_double_newline_wrapped(self, caplog: pytest.LogCaptureFixture) -> None:
        r"""Mode 2 keeps all 40 ``Cnn`` cards when the source uses ``\n\n`` between cards."""
        cards = [f"C{i:02d}".ljust(80) for i in range(1, 41)]
        wrapped = "\n\n".join(cards) + "\n"

        ds = xr.Dataset()
        with (
            patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "2"}),
            caplog.at_level(logging.WARNING, logger="mdio.ingestion.segy.file_headers"),
        ):
            result = add_segy_file_headers(ds, _segy_info(wrapped))

        stored = result["segy_file_header"].attrs["textHeader"]
        stored_rows = stored.split("\n")
        assert len(stored_rows) == 40
        for i, row in enumerate(stored_rows, start=1):
            assert row.startswith(f"C{i:02d}"), f"card {i} lost; got {row!r}"
        assert any("Correcting malformed" in record.message for record in caplog.records)
