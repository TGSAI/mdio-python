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
from segy.schema.format import TextHeaderEncoding
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


def prompt_for_header_fields(field_type: str, segy_spec: SegySpec | None = None) -> list[HeaderField]:
    """Prompt user to customize header fields with interactive choices.

    Improvements over the previous version:
    - No comma-separated input is required.
    - You can pick known fields from the current SEG-Y spec via a checkbox.
    - You can add new fields using guided prompts for name, byte, and format.
    """
    from segy.schema.header import HeaderField

    def _get_known_fields() -> list[HeaderField]:
        if segy_spec is None:
            return []
        if field_type.lower() == "binary":
            return list(getattr(segy_spec.binary.header, "fields", []))
        if field_type.lower() == "trace":
            return list(getattr(segy_spec.trace.header, "fields", []))
        return []

    def _format_choice(hf: HeaderField) -> str:
        # Show helpful info for each known field
        return f"{hf.name} (byte={hf.byte}, format={hf.format})"

    fields: list[HeaderField] = []

    if not questionary.confirm(f"Customize {field_type} header fields?", default=False).ask():
        return fields

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
            if not known:
                print("No known fields available to select. You can still add a new field.")
                continue
            choices = [_format_choice(hf) for hf in known]
            selected = questionary.checkbox(f"Select {field_type} header fields to add:", choices=choices).ask() or []
            # Map chosen strings back to HeaderField models
            lookup = {_format_choice(hf): hf for hf in known}
            for label in selected:
                hf = lookup.get(label)
                if hf is not None:
                    fields.append(HeaderField.model_validate(hf.model_dump()))

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

            # Choose data format from available ScalarType values (from segy library)
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


def create_segy_spec(
    input_path: UPath, mdio_template: AbstractDatasetTemplate, preselected_encoding: str | None = None
) -> SegySpec:
    """Create SEG-Y specification interactively."""
    from segy.standards.registry import get_segy_standard

    # Preview textual header FIRST with EBCDIC by default (before selecting SEG-Y revision)
    if preselected_encoding is None:
        text_encoding = TextHeaderEncoding.EBCDIC
        segy_spec_preview = get_segy_standard(SegyStandard.REV1)
        segy_spec_preview.text_header.encoding = text_encoding
        if segy_spec_preview.ext_text_header is not None:
            segy_spec_preview.ext_text_header.spec.encoding = text_encoding
        segy_spec_preview.endianness = None

        if questionary.confirm("Preview textual header now?", default=True).ask():
            include_ext = False
            while True:
                main_txt, ext_txt_list = _read_text_headers(input_path, segy_spec_preview)
                ext_to_show = ext_txt_list if include_ext else []
                bundle = _format_header_bundle(main_txt, ext_to_show, segy_spec_preview.text_header.encoding)
                _pager(bundle)

                # If there are extended headers, let the user decide to include them next time
                if ext_txt_list:
                    include_ext = questionary.confirm(
                        "Include extended text headers in the next preview?", default=include_ext
                    ).ask()

                # Offer saving the currently displayed content
                did_save = False
                if questionary.confirm("Save displayed header(s) to a file?", default=False).ask():
                    default_hdr_path = input_path.with_name(f"{input_path.stem}_text_header.txt")
                    out_hdr_uri = questionary.text(
                        "Filename for textual header:", default=default_hdr_path.as_posix()
                    ).ask()
                    if out_hdr_uri:
                        with UPath(out_hdr_uri).open("w") as fh:
                            fh.write(bundle)
                        print(f"Textual header saved to '{out_hdr_uri}'.")
                        did_save = True

                # If the user saved, exit the preview loop without asking to preview again
                if did_save:
                    break

                # Allow switching encoding and re-previewing
                if not questionary.confirm("Switch encoding and preview again?", default=False).ask():
                    break

                new_enc = prompt_for_text_encoding()
                text_encoding = new_enc or text_encoding
                segy_spec_preview.text_header.encoding = text_encoding
                if segy_spec_preview.ext_text_header is not None:
                    segy_spec_preview.ext_text_header.spec.encoding = text_encoding
    else:
        text_encoding = preselected_encoding

    # Now prompt for SEG-Y standard and build the final spec
    segy_standard = prompt_for_segy_standard()
    segy_spec = get_segy_standard(segy_standard)
    segy_spec.text_header.encoding = text_encoding
    if segy_standard >= 1:
        segy_spec.ext_text_header.spec.encoding = text_encoding
    segy_spec.endianness = None

    # Optionally reduce to only template-required trace headers BEFORE customizations,
    # so any user-added extra fields remain intact afterwards.
    is_minimal = questionary.confirm("Import only trace headers required by template?", default=False).ask()
    if is_minimal:
        required_fields = set(mdio_template.coordinate_names) | set(mdio_template.spatial_dimension_names)
        required_fields = required_fields | {"coordinate_scalar"}
        new_fields = [field for field in segy_spec.trace.header.fields if field.name in required_fields]
        segy_spec.trace.header.fields = new_fields

    # Now prompt for any customizations; these will be applied on top of the (possibly minimal) spec.
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

    # Preview textual header before template selection when building spec interactively
    preselected_encoding: str | None = None
    if interactive and segy_spec is None:
        preselected_encoding = _interactive_text_header_preview_select_encoding(input_path)

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
        segy_spec_obj = create_segy_spec(input_path, mdio_template_obj, preselected_encoding=preselected_encoding)
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


# --- Helpers for textual header preview ---


def _read_text_headers(input_path: UPath, segy_spec: SegySpec) -> tuple[str, list[str]]:
    """Read main and extended textual headers from a SEG-Y file using the provided spec.

    Important: Avoid SegyFile.text_header and .ext_text_header cached properties so that
    switching encodings reflects immediately. We read raw bytes and decode via the spec.
    """
    from segy import SegyFile

    sf = SegyFile(input_path.as_posix(), spec=segy_spec)

    # Read and decode the main textual header directly via the spec
    # We need to clear the cached properties
    if hasattr(sf.spec.text_header, "processor"):
        del sf.spec.text_header.processor
    if hasattr(sf, "text_header"):
        del sf.text_header

    main: str = sf.text_header

    # Read and decode extended textual headers (if present) directly via the spec
    ext: list[str] = []
    ext_spec = sf.spec.ext_text_header
    if ext_spec is not None:
        ext_buf = sf.fs.read_block(fn=sf.url, offset=ext_spec.offset, length=ext_spec.itemsize)
        ext = ext_spec.decode(ext_buf)

    return main, ext


def _pager(content: str) -> None:
    """Show content via a pager if available; fallback to plain print."""
    try:
        click.echo_via_pager(content)
    except Exception:  # pragma: no cover - fallback path
        print(content)


def _format_header_bundle(main: str, ext_list: list[str] | None, encoding: str) -> str:
    """Format textual headers nicely for display or saving."""
    lines: list[str] = [
        f"Textual Header (encoding={encoding})",
        "-" * 60,
        main.rstrip("\n"),
    ]
    if ext_list:
        for i, ext in enumerate(ext_list, start=1):
            lines.extend(
                [
                    "",
                    f"Extended Text Header #{i} (encoding={encoding})",
                    "-" * 60,
                    ext.rstrip("\n"),
                ]
            )
    return "\n".join(lines)


def _interactive_text_header_preview_select_encoding(input_path: UPath) -> str:
    """Run textual header preview before template selection and return the chosen encoding.

    Uses a temporary REV1 spec and starts with EBCDIC by default. Allows switching
    between EBCDIC/ASCII and saving the displayed headers.
    """
    from segy.standards.registry import get_segy_standard

    text_encoding = TextHeaderEncoding.EBCDIC
    segy_spec_preview = get_segy_standard(SegyStandard.REV1)
    segy_spec_preview.text_header.encoding = text_encoding
    if segy_spec_preview.ext_text_header is not None:
        segy_spec_preview.ext_text_header.spec.encoding = text_encoding
    segy_spec_preview.endianness = None

    if questionary.confirm("Preview textual header now?", default=True).ask():
        include_ext = False
        while True:
            main_txt, ext_txt_list = _read_text_headers(input_path, segy_spec_preview)
            ext_to_show = ext_txt_list if include_ext else []
            bundle = _format_header_bundle(main_txt, ext_to_show, segy_spec_preview.text_header.encoding)
            _pager(bundle)

            if ext_txt_list:
                include_ext = questionary.confirm(
                    "Include extended text headers in the next preview?", default=include_ext
                ).ask()

            did_save = False
            if questionary.confirm("Save displayed header(s) to a file?", default=False).ask():
                default_hdr_path = input_path.with_name(f"{input_path.stem}_text_header.txt")
                out_hdr_uri = questionary.text(
                    "Filename for textual header:", default=default_hdr_path.as_posix()
                ).ask()
                if out_hdr_uri:
                    with UPath(out_hdr_uri).open("w") as fh:
                        fh.write(bundle)
                    print(f"Textual header saved to '{out_hdr_uri}'.")
                    did_save = True

            # If the user saved, exit the preview loop without asking to preview again
            if did_save:
                break

            if not questionary.confirm("Switch encoding and preview again?", default=False).ask():
                break

            new_enc = prompt_for_text_encoding()
            text_encoding = new_enc or text_encoding
            segy_spec_preview.text_header.encoding = text_encoding
            if segy_spec_preview.ext_text_header is not None:
                segy_spec_preview.ext_text_header.spec.encoding = text_encoding

    return text_encoding
