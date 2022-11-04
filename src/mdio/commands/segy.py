"""SEG-Y Import/Export CLI Plugin."""


try:
    import click
    import click_params

    import mdio
except SystemError:
    pass


SEGY_HELP = """
MDIO and SEG-Y conversion utilities. Below is general information
about the SEG-Y format and MDIO features. For import or export
specific functionality check the import or export modules:

\b
mdio segy import --help
mdio segy export --help

MDIO can import SEG-Y files to a modern, chunked format.

The SEG-Y format is defined by the Society of Exploration Geophysicists
as a data transmission format and has its roots back to 1970s. There are
currently multiple revisions of the SEG-Y format.

MDIO can unravel and index any SEG-Y file that is on a regular grid.
There is no limitation to dimensionality of the data, as long as it can
be represented on a regular grid. Most seismic surveys are on a regular
grid of unique shot/receiver IDs or  are imaged on regular CDP or
INLINE/CROSSLINE grids.

The SEG-Y headers are used as identifiers to take the flattened SEG-Y
data and convert it to the multi-dimensional tensor representation. An
example of ingesting a 3-D Post-Stack seismic data can be though as the
following, per the SEG-Y Rev1 standard:

\b
--header-names inline,crossline
--header-locations 189,193
--header-lengths 4,4

\b
Our recommended chunk sizes are:
(Based on GCS benchmarks)
\b
3D: 64 x 64 x 64
2D: 512 x 512

The 4D+ datasets chunking recommendation depends on the type of
4D+ dataset (i.e. SHOT vs CDP data will have different chunking).

MDIO also import or export big and little endian coded IBM or IEEE floating
point formatted SEG-Y files. MDIO can also build a grid from arbitrary header
locations for indexing. However, the headers are stored as the SEG-Y Rev 1
after ingestion.
"""

cli = click.Group(name="segy", help=SEGY_HELP)


@cli.command(name="import")
@click.option(
    "-i",
    "--input-segy-path",
    required=True,
    help="Input SEG-Y file path.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output-mdio-file",
    required=True,
    help="Output path or URL to write the mdio file.",
    type=click.STRING,
)
@click.option(
    "-loc",
    "--header-locations",
    required=True,
    help="Byte locations of the index attributes in SEG-Y trace header.",
    type=click_params.IntListParamType(),
)
@click.option(
    "-len",
    "--header-lengths",
    required=False,
    help="Byte lengths of the index attributes in SEG-Y trace header.",
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
    "-tolerance",
    "--compression-tolerance",
    required=False,
    default=0.01,
    help="Lossy compression tolerance in ZFP.",
    type=click.FLOAT,
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
    output_mdio_file,
    header_locations,
    header_lengths,
    header_names,
    chunk_size,
    endian,
    lossless,
    compression_tolerance,
    storage_options,
    overwrite,
):
    """Ingest SEG-Y file to MDIO.

    SEG-Y format is explained in the "segy" group of the command line
    interface. To see additional information run:

    mdio segy --help

    MDIO allows ingesting flattened seismic surveys in SEG-Y format
    into a multidimensional tensor that represents the correct
    geometry of the seismic dataset.

    The SEG-Y file must be on disk, MDIO currently does not support
    reading SEG-Y directly from the cloud object store.

    The output MDIO file can be local or on the cloud. For local
    files, a UNIX or Windows path is sufficient. However, for cloud
    stores, an appropriate protocol must be provided. Some examples:

    File Path Patterns:

        \b
        If we are working locally:
        --input_segy_path local_seismic.segy
        --output-mdio-path local_seismic.mdio

        \b
        If we are working on the cloud on Amazon Web Services:
        --input_segy_path local_seismic.segy
        --output-mdio-path s3://bucket/local_seismic.mdio

        \b
        If we are working on the cloud on Google Cloud:
        --input_segy_path local_seismic.segy
        --output-mdio-path gs://bucket/local_seismic.mdio

        \b
        If we are working on the cloud on Microsoft Azure:
        --input_segy_path local_seismic.segy
        --output-mdio-path abfs://bucket/local_seismic.mdio

    The SEG-Y headers for indexing must also be specified. The
    index byte locations (starts from 1) are the minimum amount
    of information needed to index the file. However, we suggest
    giving names to the index dimensions, and if needed providing
    the header lengths if they are not standard. By default, all header
    entries are assumed to be 4-byte long.

    The chunk size depends on the data type, however, it can be
    chosen to accommodate any workflow's access patterns. See examples
    below for some common use cases.

    By default, the data is ingested with LOSSLESS compression. This
    saves disk space in the range of 20% to 40%. MDIO also allows
    data to be compressed using the ZFP compressor's fixed accuracy
    lossy compression. If lossless parameter is set to False and MDIO
    was installed using the lossy extra; then the data will be compressed
    to approximately 30% of its original size and will be perceptually
    lossless. The compression amount can be adjusted using the option
    compression_tolerance (float). Values less than 1 gives good results.
    The higher the value, the more compression, but will introduce artifacts.
    The default value is 0.01 tolerance, however we get good results
    up to 0.5; where data is almost compressed to 10% of its original size.
    NOTE: This assumes data has amplitudes normalized to have approximately
    standard deviation of 1. If dataset has values smaller than this
    tolerance, a lot of loss may occur.

    Usage:

        Below are some examples of ingesting standard SEG-Y files per
        the SEG-Y Revision 1 and 2 formats.

        \b
        3D Seismic Post-Stack:
        Chunks: 128 inlines x 128 crosslines x 128 samples
        --header-locations 189,193
        --header-names inline,crossline


        \b
        3D Seismic Imaged Pre-Stack Gathers:
        Chunks: 16 inlines x 16 crosslines x 16 offsets x 512 samples
        --header-locations 189,193,37
        --header-names inline,crossline,offset
        --chunk-size 16,16,16,512

        \b
        2D Seismic Shot Data (Byte Locations Vary):
        Chunks: 16 shots x 256 channels x 512 samples
        --header-locations 9,13
        --header-names shot,chan
        --chunk-size 16,256,512

        \b
        3D Seismic Shot Data (Byte Locations Vary):
        Let's assume streamer number is at byte 213 as
        a 2-byte integer field.
        Chunks: 8 shots x 2 cables x 256 channels x 512 samples
        --header-locations 9,213,13
        --header-names shot,cable,chan
        --header-lengths 4,2,4
        --chunk-size 8,2,256,512
    """
    mdio.segy_to_mdio(
        segy_path=input_segy_path,
        mdio_path_or_buffer=output_mdio_file,
        index_bytes=header_locations,
        index_lengths=header_lengths,
        index_names=header_names,
        chunksize=chunk_size,
        endian=endian,
        lossless=lossless,
        compression_tolerance=compression_tolerance,
        storage_options=storage_options,
        overwrite=overwrite,
    )


@cli.command(name="export")
@click.option(
    "-i",
    "--input-mdio-file",
    required=True,
    help="Input path of the mdio file",
    type=click.STRING,
)
@click.option(
    "-o",
    "--output-segy-path",
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
    input_mdio_file,
    output_segy_path,
    access_pattern,
    segy_format,
    storage_options,
    endian,
):
    """Export MDIO file to SEG-Y.

    SEG-Y format is explained in the "segy" group of the command line
    interface. To see additional information run:

    mdio segy --help

    MDIO allows exporting multidimensional seismic data back to the flattened
    seismic format SEG-Y, to be used in data transmission.

    The input headers are preserved as is, and will be transferred to the
    output file.

    The user has control over the endianness, and the floating point data
    type. However, by default we export as Big-Endian IBM float, per the
    SEG-Y format's default.

    The input MDIO can be local or cloud based. However, the output SEG-Y
    will be generated locally.
    """
    mdio.mdio_to_segy(
        mdio_path_or_buffer=input_mdio_file,
        output_segy_path=output_segy_path,
        access_pattern=access_pattern,
        out_sample_format=segy_format,
        storage_options=storage_options,
        endian=endian,
    )
