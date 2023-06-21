"""Dataset CLI Plugin."""


try:
    import click
    import click_params

    import mdio
    import json
except SystemError:
    pass


DEFAULT_HELP = """
MDIO CLI utilities. 
"""


@click.group(help=DEFAULT_HELP)
def cli():
    click.echo(f"MDIO CLI utilities")


@cli.command(name="copy")
@click.option(
    "-i",
    "--input-mdio-path",
    required=True,
    help="Input mdio path.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output-mdio-path",
    required=True,
    help="Output path or URL to write the mdio dataset.",
    type=click.STRING,
)
@click.option(
    "-exc",
    "--excludes",
    required=False,
    help="""Data to exclude during copy. i.e. chunked_012. The raw data won’t be
    copied, but it will create an empty array to be filled. If left blank, it will
    copy everything.""",
    type=click.STRING,
)
@click.option(
    "-inc",
    "--includes",
    required=False,
    help="""Data to include during copy. i.e. trace_headers. If this is not 
    specified, and certain data is excluded, it will not copy headers. If you want
    to preserve headers, specify trace_headers. If left blank, it will copy
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
    input_mdio_path,
    output_mdio_path: str,
    includes: str = "",
    excludes: str = "",
    storage_options: dict | None = None,
    overwrite: bool = False,
):
    """Copy MDIO to MDIO.
    Can also copy with empty data to be filled later. See `excludes`
    and `includes` parameters.

    More documentation about `excludes` and `includes` can be found
    in Zarr's documentation in `zarr.convenience.copy_store`.

    Args:
        source: MDIO reader or accessor instance. Data will be copied from here
        dest_path_or_buffer: Destination path. Could be any FSSpec mapping.
        excludes: Data to exclude during copy. i.e. `chunked_012`. The raw data
            won't be copied, but it will create an empty array to be filled.
            If left blank, it will copy everything.
        includes: Data to include during copy. i.e. `trace_headers`. If this is
            not specified, and certain data is excluded, it will not copy headers.
            If you want to preserve headers, specify `trace_headers`. If left blank,
            it will copy everything except specified in `excludes` parameter.
        storage_options: Storage options for the cloud storage backend.
            Default is None (will assume anonymous).
        overwrite: Overwrite destination or not.
    """
    mdio.copy_mdio(
        source=input_mdio_path,
        dest_path_or_buffer=output_mdio_path,
        excludes=excludes,
        includes=includes,
        storage_options=storage_options,
        overwrite=overwrite,
    )


@cli.command(name="info")
@click.option(
    "-i",
    "--input-mdio-file",
    required=True,
    help="Input path of the mdio file",
    type=click.STRING,
)
@click.option(
    "-format",
    "--output-format",
    required=False,
    default="plain",
    help="""Output format, plain is human readable.  JSON will output in json 
    format for easier passing. """,
    type=click.Choice(["plain", "json"]),
    show_default=True,
    show_choices=True,
)
def info(
    input_mdio_file,
    output_format,
):
    """Provide information on MDIO dataset.

    By default this returns human readable information about the grid and stats for
    the dataset. If output-format is set to json then a json is returned to
    facilitate parsing.
    """
    reader = mdio.MDIOReader(input_mdio_file, return_metadata=True)
    mdio_dict = {}
    mdio_dict["grid"] = {}
    for axis in reader.grid.dim_names:
        dim = reader.grid.select_dim(axis)
        min = dim.coords[0]
        max = dim.coords[-1]
        size = dim.coords.shape[0]
        axis_dict = {"name": axis, "min": min, "max": max, "size": size}
        mdio_dict["grid"][axis] = axis_dict

    if output_format == "plain":
        click.echo("{:<10} {:<10} {:<10} {:<10}".format("NAME", "MIN", "MAX", "SIZE"))
        click.echo("=" * 40)

        for _, axis_dict in mdio_dict["grid"].items():
            click.echo(
                "{:<10} {:<10} {:<10} {:<10}".format(
                    axis_dict["name"],
                    axis_dict["min"],
                    axis_dict["max"],
                    axis_dict["size"],
                )
            )

        click.echo("\n\n{:<10} {:<10}".format("STAT", "VALUE"))
        click.echo("=" * 20)
        for name, stat in reader.stats.items():
            click.echo("{:<10} {:<10}".format(name, stat))
    if output_format == "json":
        mdio_dict["stats"] = reader.stats
        click.echo(mdio_dict)
