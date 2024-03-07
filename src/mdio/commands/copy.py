"""MDIO Dataset copy command."""


import click_params
from click import BOOL
from click import STRING
from click import argument
from click import command
from click import option

from mdio import MDIOReader


@command(name="copy")
@argument("source-mdio-path", type=str)
@argument("target-mdio-path", type=str)
@option(
    "-access",
    "--access-pattern",
    required=False,
    default="012",
    help="Access pattern of the file",
    type=STRING,
    show_default=True,
)
@option(
    "-exc",
    "--excludes",
    required=False,
    default="",
    help="Data to exclude during copy, like `chunked_012`. The data values won’t be "
    "copied but an empty array will be created. If blank, it copies everything.",
    type=STRING,
)
@option(
    "-inc",
    "--includes",
    required=False,
    default="",
    help="Data to include during copy, like `trace_headers`. If not specified, and "
    "certain data is excluded, it will not copy headers. To preserve headers, "
    "specify trace_headers. If left blank, it will copy everything except what is "
    "specified in the 'excludes' parameter.",
    type=STRING,
)
@option(
    "-storage",
    "--storage-options",
    required=False,
    help="Custom storage options for cloud backends",
    type=click_params.JSON,
)
@option(
    "-overwrite",
    "--overwrite",
    required=False,
    default=False,
    help="Flag to overwrite if MDIO file exists",
    type=BOOL,
    show_default=True,
)
def copy(
    source_mdio_path: str,
    target_mdio_path: str,
    access_pattern: str = "012",
    includes: str = "",
    excludes: str = "",
    storage_options: dict | None = None,
    overwrite: bool = False,
):
    """Copy a MDIO dataset to anpther MDIO dataset.

    Can also copy with empty data to be filled later. See `excludes`
    and `includes` parameters.

    More documentation about `excludes` and `includes` can be found
    in Zarr's documentation in `zarr.convenience.copy_store`.
    """
    reader = MDIOReader(source_mdio_path, access_pattern=access_pattern)

    reader.copy(
        dest_path_or_buffer=target_mdio_path,
        excludes=excludes,
        includes=includes,
        storage_options=storage_options,
        overwrite=overwrite,
    )


cli = copy
