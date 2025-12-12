from __future__ import annotations

from typing import TYPE_CHECKING

import questionary
import typer
from segy.schema.format import TextHeaderEncoding
from segy.schema.header import HeaderField
from segy.schema.segy import SegyStandard
from upath import UPath

from mdio.commands.segy_helpers.text_headers import _format_text_header
from mdio.commands.segy_helpers.text_headers import _pager
from mdio.commands.segy_helpers.text_headers import _read_text_header

if TYPE_CHECKING:  # pragma: no cover
    from segy.schema.segy import SegySpec

    from mdio.builder.templates.base import AbstractDatasetTemplate

TEXT_ENCODING = TextHeaderEncoding.EBCDIC
REVISION_MAP = {
    "rev 0": SegyStandard.REV0,
    "rev 1": SegyStandard.REV1,
    "rev 2": SegyStandard.REV2,
    "rev 2.1": SegyStandard.REV21,
}


def prompt_for_segy_standard() -> SegyStandard:
    """Prompt user to select a SEG-Y standard."""
    choices = list(REVISION_MAP.keys())
    standard_str = questionary.select("Select SEG-Y standard:", choices=choices, default="rev 1").ask()
    return SegyStandard(REVISION_MAP[standard_str])


def prompt_for_text_encoding() -> TextHeaderEncoding | None:
    """Prompt user for text header encoding (returns TextHeaderEncoding)."""
    choices = [member.name for member in TextHeaderEncoding]
    choice = questionary.select("Select text header encoding:", choices=choices, default=TEXT_ENCODING).ask()
    if choice is None:
        return None
    return TextHeaderEncoding(choice)


def prompt_for_header_fields(field_type: str, segy_spec: SegySpec) -> list[HeaderField]:
    """Prompt user to customize header fields with interactive choices."""

    def _get_known_fields() -> list[HeaderField]:
        """Get known fields for the given field type."""
        if field_type.lower() == "binary":
            return segy_spec.binary.header.fields
        if field_type.lower() == "trace":
            return segy_spec.trace.header.fields
        return []

    def _format_choice(hf: HeaderField) -> str:
        """Format a header field choice for the checkbox."""
        return f"{hf.name} (byte={hf.byte}, format={hf.format})"

    if not questionary.confirm(f"Customize {field_type} header fields?", default=False).ask():
        return []

    fields = []
    while True:
        action = questionary.select(
            f"Customize {field_type} header fields — choose an action:",
            choices=[
                "Add from known fields",
                "Add a new field",
                "View current selections",
                "Clear selections",
                "Done",
            ],
            default="Add a new field",
        ).ask()

        if action == "Add from known fields":
            known = _get_known_fields()
            choices = [_format_choice(hf) for hf in known]
            selected = questionary.checkbox(f"Pick {field_type} header fields to add:", choices=choices).ask()
            lookup = {_format_choice(hf): hf for hf in known}
            for label in selected:
                header_field = lookup.get(label)
                if header_field is not None:
                    fields.append(header_field)

        elif action == "Add a new field":
            name = questionary.text("Field name (e.g., inline):").ask()
            if not name:
                print("Name cannot be empty.")
                continue

            byte_str = questionary.text("Starting byte (integer):").ask()
            try:
                byte_val = int(byte_str)
            except (TypeError, ValueError):
                print("Byte must be an integer.")
                continue

            from segy.schema.format import ScalarType

            fmt_choices = [s.value for s in ScalarType]
            format_ = questionary.select("Data format:", choices=fmt_choices, default="int32").ask()
            if not format_:
                print("Format cannot be empty.")
                continue

            try:
                valid_field = HeaderField.model_validate({"name": name, "byte": byte_val, "format": format_})
            except Exception as exc:  # pydantic validation error
                print(f"Invalid field specification: {exc}")
                continue
            fields.append(valid_field)

        elif action == "View current selections":
            if not fields:
                print("No custom fields selected yet.")
            else:
                print("Currently selected fields:")
                for i, hf in enumerate(fields, start=1):
                    print(f"  {i}. {hf.name} (byte={hf.byte}, format={hf.format})")

        elif action == "Clear selections":
            if fields and questionary.confirm("Clear all selected fields?", default=False).ask():
                fields = []

        elif action == "Done":
            break

    return fields


def prompt_for_mdio_template() -> AbstractDatasetTemplate:
    """Prompt user to select a MDIO template."""
    from mdio.builder.template_registry import get_template_registry

    registry = get_template_registry()
    choices = registry.list_all_templates()
    template_name = questionary.select("Select MDIO template:", choices=choices).ask()

    if template_name is None:
        raise typer.Abort

    return registry.get(template_name)


def interactive_text_header(input_path: UPath) -> TextHeaderEncoding:
    """Run textual header preview and return the chosen encoding."""
    from segy.standards.registry import get_segy_standard

    text_encoding = TextHeaderEncoding.EBCDIC
    segy_spec_preview = get_segy_standard(SegyStandard.REV0)
    segy_spec_preview.text_header.encoding = text_encoding
    segy_spec_preview.endianness = None

    if questionary.confirm("Preview textual header now?", default=True).ask():
        while True:
            main_txt = _read_text_header(input_path, segy_spec_preview)
            formatted_txt = _format_text_header(main_txt, segy_spec_preview.text_header.encoding)
            _pager(formatted_txt)

            did_save = False
            if questionary.confirm("Save displayed header(s) to a file?", default=False).ask():
                default_hdr_uri = input_path.with_name(f"{input_path.stem}_text_header.txt").as_posix()
                out_hdr_uri = questionary.text("Filename for text header:", default=default_hdr_uri).ask()
                if out_hdr_uri:
                    with UPath(out_hdr_uri).open("w") as fp:
                        fp.write(formatted_txt)
                    print(f"Textual header saved to '{out_hdr_uri}'.")
                    did_save = True

            if did_save:
                break

            if not questionary.confirm("Switch encoding and preview again?", default=False).ask():
                break

            new_enc = prompt_for_text_encoding()
            text_encoding = new_enc or text_encoding
            segy_spec_preview.text_header.encoding = text_encoding

    return text_encoding
