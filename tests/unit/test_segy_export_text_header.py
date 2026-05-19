"""Tests for export-side text header guarding in ``mdio.segy.creation``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from segy.factory import SegyFactory
from segy.standards import get_segy_standard

from mdio.segy.creation import _ensure_exportable_text_header

if TYPE_CHECKING:
    import pytest


def _well_formed_header() -> str:
    """Build a 40x80 header where each row reads ``Cnn ...spaces``."""
    return "\n".join([f"C{i:02d}".ljust(80) for i in range(1, 41)])


def _replacement_char_header() -> str:
    """Build a 40x80 header with U+FFFD scattered through the last three cards."""
    rows = [f"C{i:02d}".ljust(80) for i in range(1, 41)]
    rows[37] = "\ufffdC38" + " " * 76
    rows[38] = "\ufffdC39" + " " * 76
    rows[39] = "\ufffdC40 END EBCDIC" + " " * 65
    return "\n".join(rows)


class TestEnsureExportableTextHeader:
    """The export guard repairs malformed headers and warns; otherwise no-op."""

    def test_passthrough_when_well_formed(self, caplog: pytest.LogCaptureFixture) -> None:
        """Well-formed input is returned unchanged with no warning."""
        header = _well_formed_header()
        with caplog.at_level(logging.WARNING, logger="mdio.segy.creation"):
            result = _ensure_exportable_text_header(header)
        assert result == header
        assert not any("repaired" in record.message for record in caplog.records)

    def test_repairs_replacement_char_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """U+FFFD is repaired and a warning is logged."""
        with caplog.at_level(logging.WARNING, logger="mdio.segy.creation"):
            result = _ensure_exportable_text_header(_replacement_char_header())
        assert "\ufffd" not in result
        result.replace("\n", "").encode("ascii")
        assert any("repaired" in record.message for record in caplog.records)

    def test_repairs_short_layout(self, caplog: pytest.LogCaptureFixture) -> None:
        """Header with fewer than 40 cards is padded out to 40 rows of 80 chars."""
        short = "\n".join(["C01".ljust(80)] * 5)
        with caplog.at_level(logging.WARNING, logger="mdio.segy.creation"):
            result = _ensure_exportable_text_header(short)
        rows = result.split("\n")
        assert len(rows) == 40
        assert all(len(row) == 80 for row in rows)
        assert any("repaired" in record.message for record in caplog.records)

    def test_repaired_header_is_accepted_by_segy_factory(self) -> None:
        """Repair output round-trips through ``factory.create_textual_header`` to 3200 bytes."""
        spec = get_segy_standard(1.0)
        factory = SegyFactory(spec=spec, sample_interval=2000, samples_per_trace=1)

        repaired = _ensure_exportable_text_header(_replacement_char_header())
        encoded = factory.create_textual_header(repaired)

        assert len(encoded) == 3200

    def test_repairs_double_newline_wrapped(self, caplog: pytest.LogCaptureFixture) -> None:
        r"""Cards terminated with ``\n\n`` keep all 40 ``Cnn`` prefixes after repair."""
        cards = [f"C{i:02d}".ljust(80) for i in range(1, 41)]
        wrapped = "\n\n".join(cards) + "\n"

        with caplog.at_level(logging.WARNING, logger="mdio.segy.creation"):
            repaired = _ensure_exportable_text_header(wrapped)

        repaired_rows = repaired.split("\n")
        assert len(repaired_rows) == 40
        for i, row in enumerate(repaired_rows, start=1):
            assert row.startswith(f"C{i:02d}"), f"card {i} lost; got {row!r}"
        assert any("repaired" in record.message for record in caplog.records)

        spec = get_segy_standard(1.0)
        factory = SegyFactory(spec=spec, sample_interval=2000, samples_per_trace=1)
        assert len(factory.create_textual_header(repaired)) == 3200
