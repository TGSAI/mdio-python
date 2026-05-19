"""SEG-Y textual file header validation and sanitization helpers."""

from __future__ import annotations

import re

EXPECTED_ROWS = 40
EXPECTED_COLS = 80
ASCII_MAX_ORD = 127

_REPORT_LIMIT = 5
_NEWLINE_RUN = re.compile(r"\n{2,}")


def _is_safe_char(char: str) -> bool:
    """Return True if char is 7-bit ASCII and printable."""
    return ord(char) <= ASCII_MAX_ORD and char.isprintable()


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
    r"""Validate a SEG-Y textual file header is 40 rows of 80 ASCII-printable characters.

    Args:
        text_header: Decoded text header in wrapped form (40 rows of 80 chars joined by ``\n``).

    Raises:
        ValueError: If row count, row width, or any character fails the SEG-Y ASCII contract.
    """
    rows = text_header.split("\n")

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
        positions = [j for j, c in enumerate(row) if not _is_safe_char(c)]
        if positions:
            bad_chars[i] = positions

    if bad_chars:
        err = f"Invalid text header characters: non-ASCII or non-printable at {_summarize(bad_chars)}"
        raise ValueError(err)


def sanitize_text_header(text_header: str) -> str:
    r"""Coerce a SEG-Y textual file header into the 40x80 ASCII-printable card layout.

    Runs of two or more ``\n`` collapse to one (some writers terminate cards with ``\n\n``).
    Each row gets unsafe characters replaced with spaces and is padded/truncated to 80 chars.
    The result always has exactly 40 rows.

    Args:
        text_header: Decoded textual file header string.

    Returns:
        Sanitized header that satisfies :func:`validate_text_header`.
    """
    normalized = _NEWLINE_RUN.sub("\n", text_header)
    rows = normalized.split("\n")

    sanitized: list[str] = []
    for row in rows[:EXPECTED_ROWS]:
        cleaned = "".join(c if _is_safe_char(c) else " " for c in row)
        sanitized.append(cleaned[:EXPECTED_COLS].ljust(EXPECTED_COLS))

    while len(sanitized) < EXPECTED_ROWS:
        sanitized.append(" " * EXPECTED_COLS)

    return "\n".join(sanitized)
