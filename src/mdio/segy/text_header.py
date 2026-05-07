"""SEG-Y textual file header validation and sanitization helpers.

The SEG-Y standard defines the textual file header as a 3200-byte block
organized as 40 cards of 80 characters each, encoded as either ASCII or
EBCDIC. Both encodings used by ``TGSAI/segy`` ultimately require the
in-memory string to be 7-bit ASCII (``ord(c) <= 127``) before bytes can be
written. The MDIO on-disk representation is the wrapped form: 40 lines of
exactly 80 characters joined by ``"\\n"``.

When the source bytes were ingested through a lossy EBCDIC decode, MDIO
typically receives ``U+FFFD`` (``"\uFFFD"``) replacement characters and other
non-ASCII codepoints. Those characters round-trip through MDIO storage but
fail when ``segy.factory.create_textual_header`` tries to re-encode the
header to ASCII for SEG-Y export. The helpers in this module exist to detect
that situation up-front and, when requested, repair it deterministically.

Repairs are conservative: any character that is either non-ASCII
(``ord(c) > 127``) or non-printable per :py:meth:`str.isprintable` is replaced
with an ASCII space, and the card grid is forced to exactly 40 rows of 80
columns. Newlines (``"\\n"``) are treated only as row separators; other
Unicode line-break characters (``"\\v"``, ``"\\f"``, ``"\\x85"``, ``"\u2028"``,
``"\u2029"``) are treated as content and replaced rather than re-splitting
the layout. Sanitization additionally collapses runs of two or more
``"\\n"`` to one so headers that were written with ``"\\n\\n"`` between
cards are not silently truncated to half their length.
"""

from __future__ import annotations

import re

EXPECTED_ROWS = 40
EXPECTED_COLS = 80
EXPECTED_LENGTH = EXPECTED_ROWS * EXPECTED_COLS
ASCII_MAX_ORD = 127

_REPORT_LIMIT = 5
_NEWLINE_RUN = re.compile(r"\n{2,}")


def _is_safe_char(char: str) -> bool:
    """Return True if a char is safe to round-trip through SEG-Y ASCII/EBCDIC.

    A char is "safe" when it is both 7-bit ASCII (``ord <= 127``) and printable
    per :py:meth:`str.isprintable`. ASCII space passes; ``U+FFFD``, accented
    Latin characters, control characters and tabs do not.
    """
    return ord(char) <= ASCII_MAX_ORD and char.isprintable()


def _split_rows(text_header: str) -> list[str]:
    """Split a wrapped text header into rows on ``"\\n"`` only.

    Other Unicode line-break characters (``"\\v"``, ``"\\f"``, ``"\u0085"``, etc.)
    are intentionally left in place so that lossy decodes do not silently
    re-shape the card grid. They will surface as unsafe characters during
    validation and be replaced during sanitization.
    """
    return text_header.split("\n")


def _find_unsafe(row: str) -> list[int]:
    """Return positions of characters that are not :func:`_is_safe_char`."""
    return [i for i, c in enumerate(row) if not _is_safe_char(c)]


def _summarize(mapping: dict[int, list[int]], limit: int = _REPORT_LIMIT) -> str:
    """Format ``{row: [positions]}`` for an error message, capped for readability."""
    if not mapping:
        return "{}"

    items = list(mapping.items())
    head = items[:limit]
    body = ", ".join(f"row {row}: positions {positions[:limit]}" for row, positions in head)

    extra_rows = len(items) - len(head)
    if extra_rows > 0:
        body += f" (+{extra_rows} more rows)"
    return body


def validate_text_header(text_header: str) -> None:
    """Validate a SEG-Y textual file header is 40 rows of 80 ASCII-printable characters.

    Args:
        text_header: Decoded textual file header string in the wrapped form
            (40 rows of 80 characters joined by ``"\\n"``).

    Raises:
        ValueError: If the header does not split into exactly 40 rows on
            ``"\\n"``, any row is not 80 characters wide, or any character is
            not safe to encode as 7-bit ASCII (see :func:`_is_safe_char`).
    """
    rows = _split_rows(text_header)

    if len(rows) != EXPECTED_ROWS:
        err = f"Invalid text header line count: expected {EXPECTED_ROWS}, got {len(rows)}"
        raise ValueError(err)

    bad_widths = [(i, len(row)) for i, row in enumerate(rows) if len(row) != EXPECTED_COLS]
    if bad_widths:
        capped = bad_widths[:_REPORT_LIMIT]
        suffix = f" (+{len(bad_widths) - len(capped)} more)" if len(bad_widths) > len(capped) else ""
        err = f"Invalid text header line widths: expected {EXPECTED_COLS} columns; offending rows: {capped}{suffix}"
        raise ValueError(err)

    bad_chars: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        positions = _find_unsafe(row)
        if positions:
            bad_chars[i] = positions

    if bad_chars:
        err = (
            "Invalid text header characters: non-ASCII or non-printable at "
            f"{_summarize(bad_chars)}"
        )
        raise ValueError(err)


def sanitize_text_header(text_header: str) -> str:
    """Coerce a SEG-Y textual file header into the 40x80 ASCII-printable card layout.

    Pre-processing collapses runs of two or more ``"\\n"`` into one. Some SEG-Y
    writers terminate each card with ``"\\n\\n"``, which yields 80 rows on a
    naive ``split("\\n")`` and would silently drop cards 21-40 when the row
    list is sliced to 40. Collapsing runs of newlines recovers the intended
    card layout for that common case while leaving properly-wrapped headers
    untouched.

    The normalized input is then split on ``"\\n"`` and each row is independently:

    1. Stripped of unsafe characters (any non-ASCII or non-printable codepoint
       is replaced with a single ASCII space).
    2. Right-padded with spaces or truncated to exactly 80 characters.

    Rows beyond 40 are dropped. Missing rows are appended as 80-space blanks
    so the result always contains exactly 40 lines.

    Args:
        text_header: Decoded textual file header string.

    Returns:
        Sanitized header string with rows joined by ``"\\n"``. The output is
        guaranteed to satisfy :func:`validate_text_header`.
    """
    normalized = _NEWLINE_RUN.sub("\n", text_header)
    rows = _split_rows(normalized)

    sanitized: list[str] = []
    for row in rows[:EXPECTED_ROWS]:
        cleaned = "".join(c if _is_safe_char(c) else " " for c in row)
        sanitized.append(cleaned[:EXPECTED_COLS].ljust(EXPECTED_COLS))

    while len(sanitized) < EXPECTED_ROWS:
        sanitized.append(" " * EXPECTED_COLS)

    return "\n".join(sanitized)
