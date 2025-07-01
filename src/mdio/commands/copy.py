"""MDIO Dataset copy command."""

from __future__ import annotations

from click import argument
from click import command
from click import option
from click_params import JSON


@command(name="copy")
@argument("source-mdio-path", type=str)
@argument("target-mdio-path", type=str)
@option(
    "-traces",
    "--with-traces",
    is_flag=True,
    help="Flag to overwrite the MDIO file if it exists",
    show_default=True,
)
@option(
    "-headers",
    "--with-headers",
    is_flag=True,
    help="Flag to overwrite the MDIO file if it exists",
    show_default=True,
)
@option(
    "-storage-input",
    "--storage-options-input",
    required=False,
    help="Storage options for input MDIO file.",
    type=JSON,
)
@option(
    "-storage-output",
    "--storage-options-output",
    required=False,
    help="Storage options for output MDIO file.",
    type=JSON,
)
@option(
    "-overwrite",
    "--overwrite",
    is_flag=True,
    help="Flag to overwrite the MDIO file if it exists",
    show_default=True,
)
def copy(  # noqa: PLR0913
    source_mdio_path: str,
    target_mdio_path: str,
    with_traces: bool = False,
    with_headers: bool = False,
    storage_options_input: dict | None = None,
    storage_options_output: dict | None = None,
    overwrite: bool = False,
) -> None:
    """Copy an MDIO dataset to another MDIO dataset.

    This command copies an MDIO file from a source path to a target path, optionally including
    trace data, headers, or both, for all access patterns. It creates a new MDIO file at the target
    path with the same structure as the source, and selectively copies data based on the provided
    flags. The function supports custom storage options for both input and output, enabling
    compatibility with various filesystems via FSSpec.
    """
    # Lazy import to reduce CLI startup time
    from mdio.api.convenience import copy_mdio  # noqa: PLC0415

    copy_mdio(
        source_mdio_path,
        target_mdio_path,
        overwrite,
        with_traces,
        with_headers,
        storage_options_input,
        storage_options_output,
    )


cli = copy
