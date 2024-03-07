"""MDIO Dataset copy command."""


from typing import Optional

import click
import click_params

import mdio


@click.command(name="copy")
@click.argument("source-mdio-path", type=str)
@click.argument("target-mdio-path", type=str)
@click.option(
    "-access",
    "--access-pattern",
    required=False,
    default="012",
    help="Access pattern of the file",
    type=click.STRING,
    show_default=True,
)
@click.option(
    "-exc",
    "--excludes",
    required=False,
    default="",
    help="""Data to exclude during copy. i.e. chunked_012. The raw data won’t be
    copied, but it will create an empty array to be filled. If left blank, it will
    copy everything.""",
    type=click.STRING,
)
@click.option(
    "-inc",
    "--includes",
    required=False,
    default="",
    help="""Data to include during copy. i.e. trace_headers. If this is not
    specified, and certain data is excluded, it will not copy headers. To
    preserve headers, specify trace_headers. If left blank, it will copy
    everything except specified in excludes parameter.""",
    type=click.STRING,
)
@click.option(
    "-storage",
    "--storage-options",
    required=False,
    help="Custom storage options for cloud backends",
    type=click_params.JSON,
)
@click.option(
    "-overwrite",
    "--overwrite",
    required=False,
    default=False,
    help="Flag to overwrite if mdio file if it exists",
    type=click.BOOL,
    show_default=True,
)
def copy(
    source_mdio_path: str,
    target_mdio_path: str,
    access_pattern: str = "012",
    includes: str = "",
    excludes: str = "",
    storage_options: Optional[dict] = None,
    overwrite: bool = False,
):
    """Copy a MDIO dataset to anpther MDIO dataset.

    Can also copy with empty data to be filled later. See `excludes`
    and `includes` parameters.

    More documentation about `excludes` and `includes` can be found
    in Zarr's documentation in `zarr.convenience.copy_store`.
    """
    reader = mdio.MDIOReader(
        source_mdio_path, access_pattern=access_pattern, return_metadata=True
    )
    mdio.copy_mdio(
        source=reader,
        dest_path_or_buffer=target_mdio_path,
        excludes=excludes,
        includes=includes,
        storage_options=storage_options,
        overwrite=overwrite,
    )


cli = copy
