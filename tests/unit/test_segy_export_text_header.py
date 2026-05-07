"""Tests for export-side text-header guarding in ``mdio.segy.creation``.

These cover the second half of issue #814: existing MDIO stores written by an
older version of MDIO may carry a malformed text header (typically scattered
``U+FFFD`` characters from a lossy EBCDIC import). The export path must not
crash on those stores; it should repair the header and warn instead.
"""

from __future__ import annotations

import logging

import pytest
from segy.factory import SegyFactory
from segy.standards import get_segy_standard

from mdio.segy.creation import _ensure_exportable_text_header


def _well_formed_header() -> str:
    return "\n".join([f"C{i:02d}".ljust(80) for i in range(1, 41)])


def _replacement_char_header() -> str:
    rows = [f"C{i:02d}".ljust(80) for i in range(1, 41)]
    rows[37] = "\ufffdC38" + " " * 76
    rows[38] = "\ufffdC39" + " " * 76
    rows[39] = "\ufffdC40 END EBCDIC" + " " * 65
    return "\n".join(rows)


class TestEnsureExportableTextHeader:
    """The export guard repairs malformed headers and warns; otherwise no-op."""

    def test_passthrough_when_well_formed(self, caplog: pytest.LogCaptureFixture) -> None:
        header = _well_formed_header()
        with caplog.at_level(logging.WARNING, logger="mdio.segy.creation"):
            result = _ensure_exportable_text_header(header)
        assert result == header
        assert not any("repaired" in record.message for record in caplog.records)

    def test_repairs_replacement_char_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="mdio.segy.creation"):
            result = _ensure_exportable_text_header(_replacement_char_header())
        assert "\ufffd" not in result
        result.replace("\n", "").encode("ascii")  # raises if any non-ASCII char survived
        assert any("repaired" in record.message for record in caplog.records)

    def test_repairs_short_layout(self, caplog: pytest.LogCaptureFixture) -> None:
        """A header with fewer than 40 cards is padded out so export can proceed."""
        short = "\n".join(["C01".ljust(80)] * 5)
        with caplog.at_level(logging.WARNING, logger="mdio.segy.creation"):
            result = _ensure_exportable_text_header(short)
        rows = result.split("\n")
        assert len(rows) == 40
        assert all(len(row) == 80 for row in rows)
        assert any("repaired" in record.message for record in caplog.records)

    def test_repaired_header_is_accepted_by_segy_factory(self) -> None:
        """End-to-end proof that repair output is round-trippable via the SEG-Y factory.

        Regression guard for issue #814: a malformed header that previously
        crashed ``factory.create_textual_header`` must produce a 3200-byte
        textual block after going through the export guard.
        """
        spec = get_segy_standard(1.0)
        factory = SegyFactory(spec=spec, sample_interval=2000, samples_per_trace=1)

        repaired = _ensure_exportable_text_header(_replacement_char_header())
        encoded = factory.create_textual_header(repaired)

        assert len(encoded) == 3200

    def test_repairs_double_newline_wrapped(self, caplog: pytest.LogCaptureFixture) -> None:
        """Legacy stores wrapped with ``\\n\\n`` per card must export with all 40 cards intact.

        This is the second real-world malformed sample seen in the wild
        (file ``260418_A4_…``): each card is terminated with ``\\n\\n``, which
        previously caused naive splitting to lose cards 21-40 silently. The
        export guard must collapse the double newlines and emit a 3200-byte
        textual block whose 40 cards all carry their original ``Cnn`` prefix.
        """
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
