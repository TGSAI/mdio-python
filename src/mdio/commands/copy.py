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
def copy(
    source_mdio_path: str,
    target_mdio_path: str,
    with_traces: bool = False,
    with_headers: bool = False,
    storage_options_input: dict | None = None,
    storage_options_output: dict | None = None,
    overwrite: bool = False,
) -> None:
    """Copy an MDIO dataset to another MDIO dataset.

    You can also copy empty data to be filled in later. See `excludes`
    and `includes` parameters.

    More documentation about `excludes` and `includes` can be found
    in Zarr's documentation in `zarr.convenience.copy_store`.
    """
    from mdio.api.convenience import copy_mdio

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
