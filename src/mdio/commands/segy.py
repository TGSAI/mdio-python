"""SEG-Y Import/Export CLI Plugin."""


from typing import Any

from click import BOOL
from click import FLOAT
from click import STRING
from click import Choice
from click import Group
from click import Path
from click import argument
from click import option
from click_params import JSON
from click_params import IntListParamType
from click_params import StringListParamType


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
--header-types int32,int32

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

cli = Group(name="segy", help=SEGY_HELP)


@cli.command(name="import")
@argument("segy-path", type=Path(exists=True))
@argument("mdio-path", type=STRING)
@option(
    "-loc",
    "--header-locations",
    required=True,
    help="Byte locations of the index attributes in SEG-Y trace header.",
    type=IntListParamType(),
)
@option(
    "-types",
    "--header-types",
    required=False,
    help="Data types of the index attributes in SEG-Y trace header.",
    type=StringListParamType(),
)
@option(
    "-names",
    "--header-names",
    required=False,
    help="Names of the index attributes",
    type=StringListParamType(),
)
@option(
    "-chunks",
    "--chunk-size",
    required=False,
    help="Custom chunk size for bricked storage",
    type=IntListParamType(),
)
@option(
    "-endian",
    "--endian",
    required=False,
    default="big",
    help="Endianness of the SEG-Y file",
    type=Choice(["little", "big"]),
    show_default=True,
    show_choices=True,
)
@option(
    "-lossless",
    "--lossless",
    required=False,
    default=True,
    help="Toggle lossless, and perceptually lossless compression",
    type=BOOL,
    show_default=True,
)
@option(
    "-tolerance",
    "--compression-tolerance",
    required=False,
    default=0.01,
    help="Lossy compression tolerance in ZFP.",
    type=FLOAT,
    show_default=True,
)
@option(
    "-storage",
    "--storage-options",
    required=False,
    help="Custom storage options for cloud backends",
    type=JSON,
)
@option(
    "-overwrite",
    "--overwrite",
    is_flag=True,
    help="Flag to overwrite if mdio file if it exists",
    show_default=True,
)
@option(
    "-grid-overrides",
    "--grid-overrides",
    required=False,
    help="Option to add grid overrides.",
    type=JSON,
)
def segy_import(
    segy_path: str,
    mdio_path: str,
    header_locations: list[int],
    header_types: list[str],
    header_names: list[str],
    chunk_size: list[int],
    endian: str,
    lossless: bool,
    compression_tolerance: float,
    storage_options: dict[str, Any],
    overwrite: bool,
    grid_overrides: dict[str, Any],
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
    the header types if they are not standard. By default, all header
    entries are assumed to be 4-byte long (int32).

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
        --header-types int32,int16,int32
        --chunk-size 8,2,256,512

    We can override the dataset grid by the `grid_overrides` parameter.
    This allows us to ingest files that don't conform to the true
    geometry of the seismic acquisition.

    For example if we are ingesting 3D seismic shots that don't have
    a cable number and channel numbers are sequential (i.e. each cable
    doesn't start with channel number 1; we can tell MDIO to ingest
    this with the correct geometry by calculating cable numbers and
    wrapped channel numbers. Note the missing byte location and type
    for the "cable" index.


    Usage:
        3D Seismic Shot Data (Byte Locations Vary):
        Let's assume streamer number does not exist but there are
        800 channels per cable.
        Chunks: 8 shots x 2 cables x 256 channels x 512 samples
        --header-locations 9,None,13
        --header-names shot,cable,chan
        --header-types int32,None,int32
        --chunk-size 8,2,256,512
        --grid-overrides '{"ChannelWrap": True, "ChannelsPerCable": 800,
                           "CalculateCable": True}'

        \b
        If we do have cable numbers in the headers, but channels are still
        sequential (aka. unwrapped), we can still ingest it like this.
        --header-locations 9,213,13
        --header-names shot,cable,chan
        --header-types int32,int16,int32
        --chunk-size 8,2,256,512
        --grid-overrides '{"ChannelWrap":True, "ChannelsPerCable": 800}'
        \b
        For shot gathers with channel numbers and wrapped channels, no
        grid overrides are necessary.

        In cases where the user does not know if the input has unwrapped
        channels but desires to store with wrapped channel index use:
        --grid-overrides '{"AutoChannelWrap": True}'

        \b
        For cases with no well-defined trace header for indexing a NonBinned
        grid override is provided.This creates the index and attributes an
        incrementing integer to the trace for the index based on first in first
        out. For example a CDP and Offset keyed file might have a header for offset
        as real world offset which would result in a very sparse populated index.
        Instead, the following override will create a new index from 1 to N, where
        N is the number of offsets within a CDP ensemble. The index to be auto
        generated is called "trace". Note the required "chunksize" parameter in
        the grid override. This is due to the non-binned ensemble chunksize is
        irrelevant to the index dimension chunksizes and has to be specified
        in the grid override itself. Note the lack of offset, only indexing CDP,
        providing CDP header type, and chunksize for only CDP and Sample
        dimension. The chunksize for non-binned dimension is in the grid overrides
        as described above. The below configuration will yield 1MB chunks.
        \b
        --header-locations 21
        --header-names cdp
        --header-types int32
        --chunk-size 4,1024
        --grid-overrides '{"NonBinned": True, "chunksize": 64}'

        \b
        A more complicated case where you may have a 5D dataset that is not
        binned in Offset and Azimuth directions can be ingested like below.
        However, the Offset and Azimuth dimensions will be combined to "trace"
        dimension. The below configuration will yield 1MB chunks.
        \b
        --header-locations 189,193
        --header-names inline,crossline
        --header-types int32,int32
        --chunk-size 4,4,1024
        --grid-overrides '{"NonBinned": True, "chunksize": 16}'

        \b
        For dataset with expected duplicate traces we have the following
        parameterization. This will use the same logic as NonBinned with
        a fixed chunksize of 1. The other keys are still important. The
        below example allows multiple traces per receiver (i.e. reshoot).
        \b
        --header-locations 9,213,13
        --header-names shot,cable,chan
        --header-types int32,int16,int32
        --chunk-size 8,2,256,512
        --grid-overrides '{"HasDuplicates": True}'
    """
    from mdio import segy_to_mdio

    segy_to_mdio(
        segy_path=segy_path,
        mdio_path_or_buffer=mdio_path,
        index_bytes=header_locations,
        index_types=header_types,
        index_names=header_names,
        chunksize=chunk_size,
        endian=endian,
        lossless=lossless,
        compression_tolerance=compression_tolerance,
        storage_options=storage_options,
        overwrite=overwrite,
        grid_overrides=grid_overrides,
    )


@cli.command(name="export")
@argument("mdio-file", type=STRING)
@argument("segy-path", type=Path(exists=False))
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
    "-format",
    "--segy-format",
    required=False,
    default="ibm32",
    help="SEG-Y sample format",
    type=Choice(["ibm32", "ieee32"]),
    show_default=True,
    show_choices=True,
)
@option(
    "-storage",
    "--storage-options",
    required=False,
    help="Custom storage options for cloud backends.",
    type=JSON,
)
@option(
    "-endian",
    "--endian",
    required=False,
    default="big",
    help="Endianness of the SEG-Y file",
    type=Choice(["little", "big"]),
    show_default=True,
    show_choices=True,
)
def segy_export(
    mdio_file: str,
    segy_path: str,
    access_pattern: str,
    segy_format: str,
    storage_options: dict[str, Any],
    endian: str,
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
    from mdio import mdio_to_segy

    mdio_to_segy(
        mdio_path_or_buffer=mdio_file,
        output_segy_path=segy_path,
        access_pattern=access_pattern,
        out_sample_format=segy_format,
        storage_options=storage_options,
        endian=endian,
    )
