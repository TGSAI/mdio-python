"""SEG-Y Import/Export CLI Plugin."""


try:
    import click
    import click_params

    import mdio
except SystemError:
    pass


cli = click.Group(name="segy", help="Subcommand to import / export SEG-Y files.")


@cli.command(name="import", help="Imports a SEG-Y file to mdio format.")
@click.option(
    "-i",
    "--input-segy-path",
    required=True,
    help="Input SEG-Y file",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output-path",
    required=True,
    help="Output path to write the mdio file",
    type=click.STRING,
)
@click.option(
    "-loc",
    "--header-locations",
    required=True,
    help="Byte locations of the index attributes in SEG-Y trace header",
    type=click_params.IntListParamType(),
)
@click.option(
    "-len",
    "--header-lengths",
    required=False,
    help="Byte lengths of the index attributes in SEG-Y trace header",
    type=click_params.IntListParamType(),
)
@click.option(
    "-names",
    "--header-names",
    required=False,
    help="Names of the index attributes",
    type=click_params.StringListParamType(),
)
@click.option(
    "-chunks",
    "--chunk-size",
    required=False,
    help="Custom chunk size for bricked storage",
    type=click_params.IntListParamType(),
)
@click.option(
    "-endian",
    "--endian",
    required=False,
    default="big",
    help="Endianness of the SEG-Y file",
    type=click.Choice(["little", "big"]),
    show_default=True,
    show_choices=True,
)
@click.option(
    "-lossless",
    "--lossless",
    required=False,
    default=True,
    help="Toggle lossless, and perceptually lossless compression",
    type=click.BOOL,
    show_default=True,
)
@click.option(
    "-ratio",
    "--compression_ratio",
    required=False,
    default=4,
    help="Lossy compression ratio.",
    type=click.INT,
    show_default=True,
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
def segy_import(
    input_segy_path,
    output_path,
    header_locations,
    header_lengths,
    header_names,
    chunk_size,
    endian,
    lossless,
    compression_ratio,
    storage_options,
    overwrite,
):
    """SEG-Y Import CLI entrypoint."""
    mdio.segy_to_mdio(
        segy_path=input_segy_path,
        mdio_path_or_buffer=output_path,
        index_bytes=header_locations,
        index_lengths=header_lengths,
        index_names=header_names,
        chunksize=chunk_size,
        endian=endian,
        lossless=lossless,
        compression_ratio=compression_ratio,
        storage_options=storage_options,
        overwrite=overwrite,
    )


@cli.command(name="export", help="Exports a SEG-Y file from a mdio file")
@click.option(
    "-i",
    "--input-file",
    required=True,
    help="Input path of the mdio file",
    type=click.STRING,
)
@click.option(
    "-o",
    "--output-path",
    required=True,
    help="Output SEG-Y file",
    type=click.Path(exists=False),
)
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
    "-format",
    "--segy-format",
    required=False,
    default="ibm32",
    help="SEG-Y sample format",
    type=click.Choice(["ibm32", "ieee32"]),
    show_default=True,
    show_choices=True,
)
@click.option(
    "-storage",
    "--storage-options",
    required=False,
    help="Custom storage options for cloud backends.",
    type=click_params.JSON,
)
@click.option(
    "-endian",
    "--endian",
    required=False,
    default="big",
    help="Endianness of the SEG-Y file",
    type=click.Choice(["little", "big"]),
    show_default=True,
    show_choices=True,
)
def segy_export(
    input_file,
    output_path,
    access_pattern,
    segy_format,
    storage_options,
    endian,
):
    """SEG-Y Export CLI entrypoint."""
    mdio.mdio_to_segy(
        mdio_path_or_buffer=input_file,
        output_segy_path=output_path,
        access_pattern=access_pattern,
        out_sample_format=segy_format,
        storage_options=storage_options,
        endian=endian,
    )
