"""SEG-Y CLI subcommands for importing from SEG-Y to MDIO and (future) exporting back.

This sub-app is available under the main CLI as: mdio segy <command>.
Run: mdio segy --help or mdio segy import --help for usage and examples.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any

import click
import questionary
import typer
from rich import print  # noqa: A004
from segy.schema.segy import SegyStandard
from upath import UPath

from mdio.converters.exceptions import GridTraceSparsityError
from mdio.exceptions import MDIOMissingFieldError

if TYPE_CHECKING:
    from click.core import Context
    from click.core import Parameter
    from segy.schema.header import HeaderField
    from segy.schema.segy import SegySpec

    from mdio.builder.templates.base import AbstractDatasetTemplate

app = typer.Typer(help="Convert SEG-Y <-> MDIO datasets.")


REVISION_MAP = {
    "rev 0": SegyStandard.REV0,
    "rev 1": SegyStandard.REV1,
    "rev 2": SegyStandard.REV2,
    "rev 2.1": SegyStandard.REV21,
    "custom": SegyStandard.CUSTOM,
}


class UPathParamType(click.ParamType):
    """Click parser for UPath."""

    name = "Path"

    def convert(self, value: str, param: Parameter | None, ctx: Context | None) -> UPath:  # noqa: ARG002
        """Convert string path to UPath."""
        try:
            return UPath(value)
        except Exception:
            self.fail(f"{value} can't be initialized as UPath", param, ctx)


class JSONParamType(click.ParamType):
    """Click parser for JSON."""

    name = "JSON"

    def convert(self, value: str, param: Parameter | None, ctx: Context | None) -> dict[str, Any]:  # noqa: ARG002
        """Convert JSON-like string to dict."""
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            self.fail(f"{value} is not a valid json string", param, ctx)


def prompt_for_segy_standard() -> SegyStandard:
    """Prompt user to select a SEG-Y standard."""
    choices = list(REVISION_MAP.keys())
    standard_str = questionary.select("Select SEG-Y standard:", choices=choices, default="rev 1").ask()
    return SegyStandard(REVISION_MAP[standard_str])


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


SegyOutType = Annotated[UPath, typer.Argument(help="Path to the input SEG-Y file.", click_type=UPathParamType())]
MdioOutType = Annotated[UPath, typer.Argument(help="Path to the output MDIO file.", click_type=UPathParamType())]
MDIOTemplateType = Annotated[str | None, typer.Option(help="Name of the MDIO template.")]
SegySpecType = Annotated[UPath | None, typer.Option(help="Path to the SEG-Y spec file.", click_type=UPathParamType())]
StorageOptionType = Annotated[dict | None, typer.Option(help="Options for remote storage.", click_type=JSONParamType())]
OverwriteType = Annotated[bool, typer.Option(help="Overwrite the MDIO file if it exists.")]
InteractiveType = Annotated[bool, typer.Option(help="Enable interactive prompts when template or spec are missing.")]


@app.command(name="import")
def segy_import(  # noqa: PLR0913
    input_path: SegyOutType,
    output_path: MdioOutType,
    mdio_template: MDIOTemplateType = None,
    segy_spec: SegySpecType = None,
    storage_input: StorageOptionType = None,
    storage_output: StorageOptionType = None,
    overwrite: OverwriteType = False,
    interactive: InteractiveType = False,
) -> None:
    """Convert a SEG-Y file into an MDIO dataset.

    \b
    In non-interactive mode you must provide both --mdio-template and --segy-spec.
    Use --interactive to be guided through selecting a template and building a SEG-Y spec.

    \b
    Examples:
    - Non-interactive (local files):
      mdio segy import in.segy out.mdio --mdio-template PostStack3DTime --segy-spec spec.json
    - Overwrite existing output with interactive template with spec:
      mdio segy import in.segy out.mdio --segy-spec spec.json --overwrite
    - Interactive (prompts for template and spec):
      mdio segy import in.segy out.mdio --interactive

    \b
    Notes:
    - Storage options are fsspec-compatible JSON passed to --storage-input/--storage-output.
    - The command fails if output exists unless --overwrite is provided.
    """
    if storage_input is not None:
        input_path = UPath(input_path, storage_options=storage_input)

    if storage_output is not None:
        output_path = UPath(output_path, storage_options=storage_output)

    if not input_path.is_file():
        typer.secho(f"Input file '{input_path}' does not exist.", fg="red", err=True)
        raise typer.Abort from None

    if mdio_template:
        mdio_template_obj = load_mdio_template(mdio_template)
    elif interactive:
        mdio_template_obj = prompt_for_mdio_template()
    else:
        typer.secho(
            "MDIO template is required in non-interactive mode. Provide --mdio-template or use --interactive.",
            fg="red",
            err=True,
        )
        raise typer.Exit(2)

    # Load or create SEG-Y specification
    if segy_spec:
        segy_spec_obj = load_segy_spec(segy_spec)
    elif interactive:
        segy_spec_obj = create_segy_spec(input_path)
    else:
        typer.secho(
            "SEG-Y spec is required in non-interactive mode. Provide --segy-spec or use --interactive to build one.",
            fg="red",
            err=True,
        )
        raise typer.Exit(2)

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
        typer.secho(f"Output location '{output_path}' exists. Use `--overwrite` flag to overwrite.", fg="red", err=True)
        raise typer.Abort from None
    except (MDIOMissingFieldError, GridTraceSparsityError) as err:
        typer.secho(str(err), fg="red", err=True)
        raise typer.Abort from None

    print(f"SEG-Y to MDIO conversion successful: {input_path} -> {output_path}")


MdioInType = Annotated[UPath, typer.Argument(help="Path to the input MDIO file.", click_type=UPathParamType())]
SegyOutType = Annotated[UPath, typer.Argument(help="Path to the output SEG-Y file.", click_type=UPathParamType())]
StorageOptionType = Annotated[dict | None, typer.Option(help="Options for remote storage.", click_type=JSONParamType())]
OverwriteType = Annotated[bool, typer.Option(help="Overwrite the MDIO file if it exists.")]


@app.command(name="export")
def segy_export(  # noqa: PLR0913
    input_path: MdioInType,
    output_path: SegyOutType,
    segy_spec: SegySpecType = None,
    storage_input: StorageOptionType = None,
    overwrite: OverwriteType = False,
    interactive: InteractiveType = False,
) -> None:
    """Export an MDIO dataset to SEG-Y.

    \b
    Status: not yet implemented. This command currently raises NotImplementedError.

    \b
    Example (will error until implemented):
    - mdio segy export in.mdio out.segy --segy-spec spec.json
    """
    if storage_input is not None:
        input_path = UPath(input_path, storage_options=storage_input)

    msg = f"Exporting MDIO to SEG-Y is not yet supported. Args: {locals()}"
    raise NotImplementedError(msg)


if __name__ == "__main__":
    app()
