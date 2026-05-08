"""Tests for SEG-Y textual file header validation and sanitization."""

from __future__ import annotations

import pytest

from mdio.segy.text_header import EXPECTED_COLS
from mdio.segy.text_header import EXPECTED_ROWS
from mdio.segy.text_header import sanitize_text_header
from mdio.segy.text_header import validate_text_header


def _well_formed_header() -> str:
    """Build a 40x80 header where each row reads ``Cnn ...spaces``."""
    rows = [f"C{i:02d}".ljust(EXPECTED_COLS) for i in range(1, EXPECTED_ROWS + 1)]
    return "\n".join(rows)


def _replacement_char_header() -> str:
    """Build a 40x80 header with U+FFFD scattered through the last three cards."""
    rows = [f"C{i:02d}".ljust(EXPECTED_COLS) for i in range(1, EXPECTED_ROWS + 1)]
    rows[37] = "\ufffdC38" + " " * (EXPECTED_COLS - 4)
    rows[38] = "\ufffdC39" + " " * (EXPECTED_COLS - 4)
    rows[39] = "\ufffdC40 END EBCDIC" + " " * (EXPECTED_COLS - 15)
    return "\n".join(rows)


class TestValidateTextHeader:
    """Validation accepts well-formed headers and rejects anything else."""

    def test_accepts_well_formed(self) -> None:
        """Well-formed 40x80 ASCII header passes."""
        validate_text_header(_well_formed_header())

    def test_rejects_wrong_row_count(self) -> None:
        """Wrong row count raises."""
        rows = [" " * EXPECTED_COLS] * (EXPECTED_ROWS - 1)
        with pytest.raises(ValueError, match="line count"):
            validate_text_header("\n".join(rows))

    def test_rejects_wrong_column_width(self) -> None:
        """Row that is not 80 chars wide raises."""
        rows = [" " * EXPECTED_COLS] * EXPECTED_ROWS
        rows[5] = "short"
        with pytest.raises(ValueError, match="line widths"):
            validate_text_header("\n".join(rows))

    def test_rejects_non_printable_characters(self) -> None:
        """Non-printable ASCII (NUL) is rejected."""
        rows = [" " * EXPECTED_COLS] * EXPECTED_ROWS
        rows[10] = "\x00" + " " * (EXPECTED_COLS - 1)
        with pytest.raises(ValueError, match="non-ASCII or non-printable"):
            validate_text_header("\n".join(rows))

    @pytest.mark.parametrize(
        "bad_char",
        ["\ufffd", "\xa0", "\u00e9", "\u00c1"],
        ids=["U+FFFD", "U+00A0", "U+00E9", "U+00C1"],
    )
    def test_rejects_non_ascii_printable_characters(self, bad_char: str) -> None:
        """Non-ASCII codepoints are rejected even when isprintable() is True."""
        rows = [" " * EXPECTED_COLS] * EXPECTED_ROWS
        rows[0] = bad_char + " " * (EXPECTED_COLS - 1)
        with pytest.raises(ValueError, match="non-ASCII or non-printable"):
            validate_text_header("\n".join(rows))

    def test_rejects_replacement_char_header(self) -> None:
        """Header laced with U+FFFD is rejected."""
        with pytest.raises(ValueError, match="non-ASCII or non-printable"):
            validate_text_header(_replacement_char_header())

    def test_does_not_split_on_unicode_line_separators(self) -> None:
        r"""``\v``, ``\f``, ``\x85`` are content, not row breaks."""
        rows = [" " * EXPECTED_COLS] * EXPECTED_ROWS
        rows[0] = "\x0b" + " " * (EXPECTED_COLS - 1)
        with pytest.raises(ValueError, match="non-ASCII or non-printable"):
            validate_text_header("\n".join(rows))

    def test_rejects_double_newline_wrapped(self) -> None:
        r"""Strict validation does not collapse ``\n\n``; only sanitize does."""
        rows = [f"C{i:02d}".ljust(EXPECTED_COLS) for i in range(1, EXPECTED_ROWS + 1)]
        with pytest.raises(ValueError, match="line count"):
            validate_text_header("\n\n".join(rows))

    def test_error_message_is_capped(self) -> None:
        """Pathologically broken header produces a bounded error message."""
        rows = ["\ufffd" * EXPECTED_COLS for _ in range(EXPECTED_ROWS)]
        with pytest.raises(ValueError, match="Invalid text header characters") as exc_info:
            validate_text_header("\n".join(rows))
        message = str(exc_info.value)
        assert "more rows" in message
        assert len(message) < 1000


class TestSanitizeTextHeader:
    """Sanitization replaces non-printable chars and forces 40x80 layout."""

    def test_passthrough_well_formed(self) -> None:
        """Well-formed input round-trips unchanged."""
        header = _well_formed_header()
        assert sanitize_text_header(header) == header

    def test_replaces_non_printable_with_space(self) -> None:
        """NUL/BEL bytes are replaced with spaces."""
        rows = [" " * EXPECTED_COLS] * EXPECTED_ROWS
        rows[0] = "C01\x00\x07" + " " * (EXPECTED_COLS - 5)
        cleaned = sanitize_text_header("\n".join(rows))

        cleaned_rows = cleaned.split("\n")
        assert len(cleaned_rows) == EXPECTED_ROWS
        assert cleaned_rows[0].startswith("C01  ")
        assert all(c.isprintable() for row in cleaned_rows for c in row)

    def test_replaces_replacement_char_with_space(self) -> None:
        """U+FFFD is repaired to spaces; surrounding text survives."""
        cleaned = sanitize_text_header(_replacement_char_header())
        assert "\ufffd" not in cleaned
        cleaned_rows = cleaned.split("\n")
        assert cleaned_rows[37].startswith(" C38")
        assert cleaned_rows[38].startswith(" C39")
        assert cleaned_rows[39].startswith(" C40 END EBCDIC")

    def test_replaces_unicode_line_separator_with_space(self) -> None:
        r"""``\v`` is replaced with a space, not split as a row break."""
        rows = [" " * EXPECTED_COLS] * EXPECTED_ROWS
        rows[0] = "\x0b" + " " * (EXPECTED_COLS - 1)
        cleaned = sanitize_text_header("\n".join(rows))
        cleaned_rows = cleaned.split("\n")
        assert len(cleaned_rows) == EXPECTED_ROWS
        assert cleaned_rows[0] == " " * EXPECTED_COLS

    def test_pads_short_rows_to_eighty_columns(self) -> None:
        """Short rows are right-padded with spaces."""
        rows = ["short"] * EXPECTED_ROWS
        cleaned = sanitize_text_header("\n".join(rows))

        for row in cleaned.split("\n"):
            assert len(row) == EXPECTED_COLS

    def test_truncates_long_rows_to_eighty_columns(self) -> None:
        """Rows longer than 80 chars are truncated."""
        long_row = "X" * (EXPECTED_COLS + 20)
        cleaned = sanitize_text_header("\n".join([long_row] * EXPECTED_ROWS))

        for row in cleaned.split("\n"):
            assert len(row) == EXPECTED_COLS
            assert row == "X" * EXPECTED_COLS

    def test_pads_missing_rows_with_blank_lines(self) -> None:
        """Headers with fewer than 40 rows are padded with blank lines."""
        rows = [f"C{i:02d}".ljust(EXPECTED_COLS) for i in range(1, 5)]
        cleaned = sanitize_text_header("\n".join(rows))

        cleaned_rows = cleaned.split("\n")
        assert len(cleaned_rows) == EXPECTED_ROWS
        assert cleaned_rows[-1] == " " * EXPECTED_COLS

    def test_truncates_excess_rows(self) -> None:
        """Headers with more than 40 rows are truncated to 40."""
        rows = [f"C{i:02d}".ljust(EXPECTED_COLS) for i in range(1, EXPECTED_ROWS + 5)]
        cleaned = sanitize_text_header("\n".join(rows))

        cleaned_rows = cleaned.split("\n")
        assert len(cleaned_rows) == EXPECTED_ROWS
        assert cleaned_rows[-1].startswith("C40")

    def test_output_passes_validation(self) -> None:
        """Sanitize output is always accepted by validate."""
        rows = [f"C{i:02d}\x00\x01\ufffd garbage" for i in range(1, EXPECTED_ROWS + 10)]
        cleaned = sanitize_text_header("\n".join(rows))
        validate_text_header(cleaned)

    def test_sanitized_header_is_ascii_encodable(self) -> None:
        """Sanitized output encodes as ASCII (the SEG-Y export requirement)."""
        cleaned = sanitize_text_header(_replacement_char_header())
        cleaned.replace("\n", "").encode("ascii")

    def test_collapses_double_newline_separator(self) -> None:
        r"""Headers with ``\n\n`` between cards keep all 40 cards."""
        cards = [f"C{i:02d}".ljust(EXPECTED_COLS) for i in range(1, EXPECTED_ROWS + 1)]
        wrapped = "\n\n".join(cards) + "\n"
        cleaned = sanitize_text_header(wrapped)

        cleaned_rows = cleaned.split("\n")
        assert len(cleaned_rows) == EXPECTED_ROWS
        for i, row in enumerate(cleaned_rows, start=1):
            assert row.startswith(f"C{i:02d}"), f"card {i} lost; got {row!r}"
        validate_text_header(cleaned)

    def test_collapses_runs_longer_than_two(self) -> None:
        r"""Triple (or longer) newline runs collapse to a single ``\n``."""
        cards = [f"C{i:02d}".ljust(EXPECTED_COLS) for i in range(1, EXPECTED_ROWS + 1)]
        cleaned = sanitize_text_header("\n\n\n".join(cards))

        cleaned_rows = cleaned.split("\n")
        assert len(cleaned_rows) == EXPECTED_ROWS
        assert cleaned_rows[-1].startswith("C40")
