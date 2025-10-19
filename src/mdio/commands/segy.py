"""SEG-Y CLI subcommands for import/export etc."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Annotated

import click
import questionary
import typer
from rich import print  # noqa: A004
from upath import UPath

from mdio.converters.exceptions import GridTraceSparsityError
from mdio.exceptions import MDIOMissingFieldError

if TYPE_CHECKING:
    from click.core import Context
    from click.core import Parameter
    from segy.schema.header import HeaderField
    from segy.schema.segy import SegySpec

    from mdio.builder.templates.base import AbstractDatasetTemplate

app = typer.Typer()


class UPathParam(click.ParamType):
    """Click parser for UPath."""

    name = "UPath"

    def convert(self, value: str, param: Parameter | None, ctx: Context | None) -> UPath:  # noqa: ARG002
        """Convert CLI value to UPath."""
        return UPath(value)


def prompt_for_segy_standard() -> float:
    """Prompt user to select a SEG-Y standard."""
    from segy.standards.registry import segy_standard_registry

    choices = [str(key) for key in segy_standard_registry]
    return float(questionary.select("Select SEG-Y standard:", choices=choices, default="1.0").ask())


def prompt_for_text_encoding() -> str:
    """Prompt user for text header encoding."""
    return questionary.select("Select text header encoding:", choices=["ebcdic", "ascii"], default="ebcdic").ask()


def prompt_for_header_fields(field_type: str) -> list[HeaderField]:
    """Prompt user to customize header fields."""
    from segy.schema.header import HeaderField

    fields = []
    if questionary.confirm(f"Customize {field_type} header fields?", default=False).ask():
        while True:
            custom_field = questionary.text("Enter field spec (name,byte,format), e.g., 'inline,181,int32':").ask()
            try:
                name, byte, format_ = custom_field.split(",")
                field_dict = {"name": name, "byte": int(byte), "format": format_}
                valid_field = HeaderField.model_validate(field_dict)
                fields.append(valid_field)
            except ValueError:
                print(f"Invalid {field_type} field spec '{custom_field}'. Use format: name,byte,format")
                continue
            if questionary.confirm("Done adding fields?", default=True).ask():
                break
    return fields


def create_segy_spec(input_path: UPath) -> SegySpec:
    """Create SEG-Y specification interactively."""
    from segy.standards.registry import get_segy_standard

    segy_standard = prompt_for_segy_standard()
    text_encoding = prompt_for_text_encoding()
    segy_spec = get_segy_standard(segy_standard)
    segy_spec.text_header.encoding = text_encoding
    if segy_standard >= 1:
        segy_spec.ext_text_header.spec.encoding = text_encoding
    segy_spec.endianness = None

    binary_fields = prompt_for_header_fields("binary")
    trace_fields = prompt_for_header_fields("trace")
    if binary_fields or trace_fields:
        segy_spec = segy_spec.customize(binary_header_fields=binary_fields, trace_header_fields=trace_fields)

    should_save = questionary.confirm("Save SEG-Y specification?", default=True).ask()
    if should_save:
        import json

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


def prompt_for_mdio_template() -> AbstractDatasetTemplate:
    """Prompt user to select a MDIO template."""
    from mdio.builder.template_registry import get_template_registry

    registry = get_template_registry()
    choices = registry.list_all_templates()
    template_name = questionary.select("Select MDIO template:", choices=choices).ask()

    if template_name is None:
        raise typer.Abort

    return registry.get(template_name)


def load_mdio_template(mdio_template_name: str) -> AbstractDatasetTemplate:
    """Load MDIO template from registry or select one interactively."""
    from mdio.builder.template_registry import get_template_registry

    registry = get_template_registry()
    try:
        return registry.get(mdio_template_name)
    except KeyError:
        print(f"MDIO template '{mdio_template_name}' not found.")
        raise typer.Abort from None


def load_segy_spec(segy_spec_path: UPath) -> SegySpec:
    """Load SEG-Y specification from a file."""
    from pydantic import ValidationError
    from segy.schema.segy import SegySpec

    try:
        with segy_spec_path.open("r") as f:
            return SegySpec.model_validate_json(f.read())
    except FileNotFoundError:
        print(f"SEG-Y specification file '{segy_spec_path}' does not exist.")
        raise typer.Abort from None
    except ValidationError:
        print(f"Invalid SEG-Y specification file '{segy_spec_path}'.")
        raise typer.Abort from None


SegyInType = Annotated[UPath, typer.Argument(help="Path to the input SEG-Y file.", click_type=UPathParam())]
MdioOutType = Annotated[UPath, typer.Argument(help="Path to the output MDIO file.", click_type=UPathParam())]
MDIOTemplateType = Annotated[str | None, typer.Option(help="Name of the MDIO template.")]
SegySpecType = Annotated[UPath | None, typer.Option(help="Path to the SEG-Y spec file.", click_type=UPathParam())]
OverwriteType = Annotated[bool, typer.Option(help="Overwrite the MDIO file if it exists.")]


@app.command(name="import")
def segy_import(
    input_path: Annotated[UPath, typer.Argument(help="Path to the input SEG-Y file.", click_type=UPathParam())],
    output_path: Annotated[UPath, typer.Argument(help="Path to the output MDIO file.", click_type=UPathParam())],
    mdio_template: MDIOTemplateType = None,
    segy_spec: SegySpecType = None,
    overwrite: OverwriteType = False,
) -> None:
    """Import SEG-Y file to MDIO format."""
    if not input_path.is_file():
        print(f"Input file '{input_path}' does not exist.")
        raise typer.Abort from None

    mdio_template_obj = load_mdio_template(mdio_template) if mdio_template else prompt_for_mdio_template()

    # Load or create SEG-Y specification
    segy_spec_obj = load_segy_spec(segy_spec) if segy_spec else create_segy_spec(input_path)

    # Perform conversion
    from mdio.converters import segy_to_mdio

    try:
        segy_to_mdio(
            segy_spec=segy_spec_obj,
            mdio_template=mdio_template_obj,
            input_path=input_path,
            output_path=output_path,
            overwrite=overwrite,
        )
    except FileExistsError:
        print(f"Output location '{output_path}' exists. Use `--overwrite` flag to overwrite.")
        raise typer.Abort from None
    except (MDIOMissingFieldError, GridTraceSparsityError) as err:
        print(err)
        raise typer.Abort from None

    print(f"SEG-Y to MDIO conversion successful: {input_path} -> {output_path}")


if __name__ == "__main__":
    app()
