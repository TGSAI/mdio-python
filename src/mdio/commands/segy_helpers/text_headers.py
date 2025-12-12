from __future__ import annotations

from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from segy.schema.format import TextHeaderEncoding
    from segy.schema.segy import SegySpec
    from upath import UPath


def _read_text_header(input_path: UPath, segy_spec: SegySpec) -> str:
    """Read file textual header from a SEG-Y file using the provided spec.

    Important: Avoid SegyFile.text_header cached properties so that switching encodings reflects immediately.
    """
    from segy import SegyFile

    segy_file = SegyFile(input_path.as_posix(), spec=segy_spec)

    # Clear any cached instances.
    if hasattr(segy_file.spec.text_header, "processor"):
        del segy_file.spec.text_header.processor
    if hasattr(segy_file, "text_header"):
        del segy_file.text_header

    return segy_file.text_header


def _pager(content: str) -> None:
    """Show content via a pager if available; fallback to plain print."""
    try:
        typer.echo_via_pager(content)
    except Exception:
        print(content)


def _format_text_header(main: str, encoding: TextHeaderEncoding) -> str:
    """Format textual headers nicely for display or saving."""
    enc_label = getattr(encoding, "value", str(encoding))
    lines: list[str] = [
        f"Textual Header (encoding={enc_label})",
        "-" * 60,
        main.rstrip("\n"),
    ]
    return "\n".join(lines)
