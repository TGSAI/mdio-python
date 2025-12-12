from __future__ import annotations

import json
from typing import TYPE_CHECKING

import questionary
import typer
from rich import print  # noqa: A004
from upath import UPath

from mdio.commands.segy_helpers.interactive import _interactive_text_header_preview_select_encoding
from mdio.commands.segy_helpers.interactive import prompt_for_header_fields
from mdio.commands.segy_helpers.interactive import prompt_for_segy_standard

if TYPE_CHECKING:
    from segy.schema.format import TextHeaderEncoding
    from segy.schema.segy import SegySpec

    from mdio.builder.templates.base import AbstractDatasetTemplate


def load_mdio_template(mdio_template_name: str) -> AbstractDatasetTemplate:
    """Load MDIO template from registry or fail with Typer.Abort."""
    from mdio.builder.template_registry import get_template_registry

    registry = get_template_registry()
    try:
        return registry.get(mdio_template_name)
    except KeyError:
        typer.secho(f"MDIO template '{mdio_template_name}' not found.", fg="red", err=True)
        raise typer.Abort from None


def load_segy_spec(segy_spec_path: UPath) -> SegySpec:
    """Load SEG-Y specification from a file."""
    from pydantic import ValidationError
    from segy.schema.segy import SegySpec

    try:
        with segy_spec_path.open("r") as f:
            return SegySpec.model_validate_json(f.read())
    except FileNotFoundError:
        typer.secho(f"SEG-Y specification file '{segy_spec_path}' does not exist.", fg="red", err=True)
        raise typer.Abort from None
    except ValidationError:
        typer.secho(f"Invalid SEG-Y specification file '{segy_spec_path}'.", fg="red", err=True)
        raise typer.Abort from None


def create_segy_spec(
    input_path: UPath, mdio_template: AbstractDatasetTemplate, preselected_encoding: TextHeaderEncoding | None = None
) -> SegySpec:
    """Create SEG-Y specification interactively."""
    from segy.standards.registry import get_segy_standard

    # Preview textual header FIRST with EBCDIC by default (before selecting SEG-Y revision)
    if preselected_encoding is None:
        text_encoding = _interactive_text_header_preview_select_encoding(input_path)
    else:
        text_encoding = preselected_encoding

    # Now prompt for SEG-Y standard and build the final spec
    segy_standard = prompt_for_segy_standard()
    segy_spec = get_segy_standard(segy_standard)
    segy_spec.text_header.encoding = text_encoding
    if segy_standard >= 1:
        segy_spec.ext_text_header.spec.encoding = text_encoding
    segy_spec.endianness = None

    # Optionally reduce to only template-required trace headers
    is_minimal = questionary.confirm("Import only trace headers required by template?", default=False).ask()
    if is_minimal:
        required_fields = set(mdio_template.coordinate_names) | set(mdio_template.spatial_dimension_names)
        required_fields = required_fields | {"coordinate_scalar"}
        new_fields = [field for field in segy_spec.trace.header.fields if field.name in required_fields]
        segy_spec.trace.header.fields = new_fields

    # Prompt for any customizations
    binary_fields = prompt_for_header_fields("binary", segy_spec)
    trace_fields = prompt_for_header_fields("trace", segy_spec)
    if binary_fields or trace_fields:
        segy_spec = segy_spec.customize(binary_header_fields=binary_fields, trace_header_fields=trace_fields)

    should_save = questionary.confirm("Save SEG-Y specification?", default=True).ask()
    if should_save:
        from segy import SegyFile

        out_segy_spec_path = input_path.with_name(f"{input_path.stem}_segy_spec.json")
        out_segy_spec_uri = out_segy_spec_path.as_posix()

        custom_uri = questionary.text("Filename for SEG-Y Specification:", default=out_segy_spec_uri).ask()
        custom_path = UPath(custom_uri)
        updated_spec = SegyFile(input_path.as_posix(), spec=segy_spec).spec

        with custom_path.open(mode="w") as f:
            json.dump(updated_spec.model_dump(mode="json"), f, indent=2)
        print(f"SEG-Y specification saved to '{custom_uri}'.")

    return segy_spec
