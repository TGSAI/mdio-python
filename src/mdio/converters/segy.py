"""Conversion from SEG-Y to MDIO."""


from __future__ import annotations

import logging
import os
from datetime import datetime
from datetime import timezone
from importlib import metadata
from typing import Any
from typing import Sequence

import numpy as np
import segyio
import zarr

from mdio.api.io_utils import process_url
from mdio.converters.exceptions import EnvironmentFormatError
from mdio.converters.exceptions import GridTraceCountError
from mdio.converters.exceptions import GridTraceSparsityError
from mdio.core import Grid
from mdio.core.utils_write import write_attribute
from mdio.segy import blocked_io
from mdio.segy.byte_utils import Dtype
from mdio.segy.helpers_segy import create_zarr_hierarchy
from mdio.segy.parsers import parse_binary_header
from mdio.segy.parsers import parse_text_header
from mdio.segy.utilities import get_grid_plan


logger = logging.getLogger(__name__)

try:
    API_VERSION = metadata.version("multidimio")
except metadata.PackageNotFoundError:
    API_VERSION = "unknown"

BACKENDS = ["s3", "gcs", "gs", "az", "abfs"]


def parse_index_types(
    str_types: Sequence[str] | None, num_index: int
) -> Sequence[Dtype]:
    """Convert string type keys to Dtype enums."""
    if str_types is None:
        parsed_types = [Dtype.INT32] * num_index
    else:
        try:
            parsed_types = [Dtype[_type.upper()] for _type in str_types]
        except KeyError as exc:
            msg = (
                "Unsupported header data-type. 'index_types' must be in "
                f"{list(Dtype.__members__.keys())}"
            )
            raise KeyError(msg) from exc

    return parsed_types


def grid_density_qc(grid: Grid, num_traces: int) -> None:
    """QC for sensible Grid density.

    Basic qc of the grid to check density and provide warning/exception
    when indexing is problematic to provide user with insights to the use.
    If trace density on the specified grid is less than 50% a warning is
    logged.  If density is less than 10% an exception is raised. To ignore
    trace sparsity check set environment variable:
        MDIO_IGNORE_CHECKS = True
    To change the ratio set the environment variable:
        MDIO__GRID__SPARSITY_RATIO_LIMIT = 10

    Args:
        grid: The grid instance to check.
        num_traces: Expected number of traces.

    Raises:
        GridTraceSparsityError: Raised if the grid is significantly larger
            than the number of traces in the SEG-Y file. By default the error
            is raised if the grid is more than 10 times larger than the number
            of traces in the SEG-Y file. This can be disabled by setting the
            environment variable `MDIO_IGNORE_CHECKS` to `True`. The limit can
            be changed by setting the environment variable
            `MDIO__GRID__SPARSITY_RATIO_LIMIT`.
        EnvironmentFormatError: Raised if the environment variable
            MDIO__GRID__SPARSITY_RATIO_LIMIT is not a float.
    """
    grid_traces = np.prod(grid.shape[:-1], dtype=np.uint64)  # Exclude sample
    dims = {k: v for k, v in zip(grid.dim_names, grid.shape)}  # noqa: B905

    logger.debug(f"Dimensions: {dims}")
    logger.debug(f"num_traces = {num_traces}")
    logger.debug(f"grid_traces = {grid_traces}")
    logger.debug(f"sparsity = {grid_traces / num_traces}")

    grid_sparsity_ratio_limit = os.getenv("MDIO__GRID__SPARSITY_RATIO_LIMIT", 10)
    try:
        grid_sparsity_ratio_limit_ = float(grid_sparsity_ratio_limit)
    except ValueError:
        raise EnvironmentFormatError(
            "MDIO__GRID__SPARSITY_RATIO_LIMIT", "float"
        ) from None

    # Warning if we have above 50% sparsity.
    msg = ""
    if grid_traces > min(2, grid_sparsity_ratio_limit_) * num_traces:
        msg = (
            f"Proposed ingestion grid is sparse. Ingestion grid: {dims}. "
            f"SEG-Y trace count:{num_traces}, grid trace count: {grid_traces}."
        )
        for dim_name in grid.dim_names:
            dim_min = grid.get_min(dim_name)
            dim_max = grid.get_max(dim_name)
            msg += f"\n{dim_name} min: {dim_min} max: {dim_max}"

        logger.warning(msg)

    # Extreme case where the grid is very sparse (usually user error)
    if grid_traces > grid_sparsity_ratio_limit_ * num_traces:
        logger.warning("WARNING: Sparse mdio grid detected!")
        if os.getenv("MDIO__IGNORE_CHECKS", False):
            # Do not raise an exception if MDIO_IGNORE_CHECK is False
            pass
        else:
            raise GridTraceSparsityError(grid.shape, num_traces, msg)


def segy_to_mdio(
    segy_path: str,
    mdio_path_or_buffer: str,
    index_bytes: Sequence[int],
    index_names: Sequence[str] | None = None,
    index_types: Sequence[str] | None = None,
    chunksize: Sequence[int] | None = None,
    endian: str = "big",
    lossless: bool = True,
    compression_tolerance: float = 0.01,
    storage_options: dict[str, Any] | None = None,
    overwrite: bool = False,
    grid_overrides: dict | None = None,
) -> None:
    """Convert SEG-Y file to MDIO format.

    MDIO allows ingesting flattened seismic surveys in SEG-Y format into a
    multidimensional tensor that represents the correct geometry of the
    seismic dataset.

    The SEG-Y file must be on disk, MDIO currently does not support reading
    SEG-Y directly from the cloud object store.

    The output MDIO file can be local or on the cloud. For local files, a
    UNIX or Windows path is sufficient. However, for cloud stores, an
    appropriate protocol must be provided. See examples for more details.

    The SEG-Y headers for indexing must also be specified. The index byte
    locations (starts from 1) are the minimum amount of information needed
    to index the file. However, we suggest giving names to the index
    dimensions, and if needed providing the header lengths if they are not
    standard. By default, all header entries are assumed to be 4-byte long.

    The chunk size depends on the data type, however, it can be chosen to
    accommodate any workflow's access patterns. See examples below for some
    common use cases.

    By default, the data is ingested with LOSSLESS compression. This saves
    disk space in the range of 20% to 40%. MDIO also allows data to be
    compressed using the ZFP compressor's fixed rate lossy compression. If
    lossless parameter is set to False and MDIO was installed using the lossy
    extra; then the data will be compressed to approximately 30% of its
    original size and will be perceptually lossless. The compression ratio
    can be adjusted using the option compression_ratio (integer). Higher
    values will compress more, but will introduce artifacts.

    Args:
        segy_path: Path to the input SEG-Y file
        mdio_path_or_buffer: Output path for MDIO file
        index_bytes: Tuple of the byte location for the index attributes
        index_names: Tuple of the index names for the index attributes
        index_types: Tuple of the data-types for the index attributes.
            Must be in {"int16, int32, float16, float32, ibm32"}
            Default is 4-byte integers for each index key.
        chunksize : Override default chunk size, which is (64, 64, 64) if
            3D, and (512, 512) for 2D.
        endian: Endianness of the input SEG-Y. Rev.2 allows little endian.
            Default is 'big'. Must be in `{"big", "little"}`
        lossless: Lossless Blosc with zstandard, or ZFP with fixed precision.
        compression_tolerance: Tolerance ZFP compression, optional. The fixed
            accuracy mode in ZFP guarantees there won't be any errors larger
            than this value. The default is 0.01, which gives about 70%
            reduction in size. Will be ignored if `lossless=True`.
        storage_options: Storage options for the cloud storage backend.
            Default is `None` (will assume anonymous)
        overwrite: Toggle for overwriting existing store
        grid_overrides: Option to add grid overrides. See examples.

    Raises:
        GridTraceCountError: Raised if grid won't hold all traces in the
            SEG-Y file.
        ValueError: If length of chunk sizes don't match number of dimensions.
        NotImplementedError: If can't determine chunking automatically for 4D+.

    Examples:
        If we are working locally and ingesting a 3D post-stack seismic file,
        we can use the following example. This will ingest with default chunks
        of 128 x 128 x 128.

        >>> from mdio import segy_to_mdio
        >>>
        >>>
        >>> segy_to_mdio(
        ...     segy_path="prefix1/file.segy",
        ...     mdio_path_or_buffer="prefix2/file.mdio",
        ...     index_bytes=(189, 193),
        ...     index_names=("inline", "crossline")
        ... )

        If we are on Amazon Web Services, we can do it like below. The
        protocol before the URL can be `s3` for AWS, `gcs` for Google
        Cloud, and `abfs` for Microsoft Azure. In this example we also
        change the chunk size as a demonstration.

        >>> segy_to_mdio(
        ...     segy_path="prefix/file.segy",
        ...     mdio_path_or_buffer="s3://bucket/file.mdio",
        ...     index_bytes=(189, 193),
        ...     index_names=("inline", "crossline"),
        ...     chunksize=(64, 64, 512),
        ... )

        Another example of loading a 4D seismic such as 3D seismic
        pre-stack gathers is below. This will allow us to extract offset
        planes efficiently or run things in a local neighborhood very
        efficiently.

        >>> segy_to_mdio(
        ...     segy_path="prefix/file.segy",
        ...     mdio_path_or_buffer="s3://bucket/file.mdio",
        ...     index_bytes=(189, 193, 37),
        ...     index_names=("inline", "crossline", "offset"),
        ...     chunksize=(16, 16, 16, 512),
        ... )

        We can override the dataset grid by the `grid_overrides` parameter.
        This allows us to ingest files that don't conform to the true
        geometry of the seismic acquisition.

        For example if we are ingesting 3D seismic shots that don't have
        a cable number and channel numbers are sequential (i.e. each cable
        doesn't start with channel number 1; we can tell MDIO to ingest
        this with the correct geometry by calculating cable numbers and
        wrapped channel numbers. Note the missing byte location and word
        length for the "cable" index.

        >>> segy_to_mdio(
        ...     segy_path="prefix/shot_file.segy",
        ...     mdio_path_or_buffer="s3://bucket/shot_file.mdio",
        ...     index_bytes=(17, None, 13),
        ...     index_lengths=(4, None, 4),
        ...     index_names=("shot", "cable", "channel"),
        ...     chunksize=(8, 2, 128, 1024),
        ...     grid_overrides={
        ...         "ChannelWrap": True, "ChannelsPerCable": 800,
        ...         "CalculateCable": True
        ...     },
        ... )

        If we do have cable numbers in the headers, but channels are still
        sequential (aka. unwrapped), we can still ingest it like this.

        >>> segy_to_mdio(
        ...     segy_path="prefix/shot_file.segy",
        ...     mdio_path_or_buffer="s3://bucket/shot_file.mdio",
        ...     index_bytes=(17, 137, 13),
        ...     index_lengths=(4, 2, 4),
        ...     index_names=("shot_point", "cable", "channel"),
        ...     chunksize=(8, 2, 128, 1024),
        ...     grid_overrides={"ChannelWrap": True, "ChannelsPerCable": 800},
        ... )

        For shot gathers with channel numbers and wrapped channels, no
        grid overrides are necessary.

        In cases where the user does not know if the input has unwrapped
        channels but desires to store with wrapped channel index use:
        >>>    grid_overrides={"AutoChannelWrap": True,
                               "AutoChannelTraceQC":  1000000}

        For ingestion of pre-stack streamer data where the user needs to
        access/index *common-channel gathers* (single gun) then the following
        strategy can be used to densely ingest while indexing on gun number:

        >>> segy_to_mdio(
        ...     segy_path="prefix/shot_file.segy",
        ...     mdio_path_or_buffer="s3://bucket/shot_file.mdio",
        ...     index_bytes=(133, 171, 17, 137, 13),
        ...     index_lengths=(2, 2, 4, 2, 4),
        ...     index_names=("shot_line", "gun", "shot_point", "cable", "channel"),
        ...     chunksize=(1, 1, 8, 1, 128, 1024),
        ...     grid_overrides={
        ...         "AutoShotWrap": True,
        ...         "AutoChannelWrap": True,
        ...         "AutoChannelTraceQC":  1000000
        ...     },
        ... )

        For AutoShotWrap and AutoChannelWrap to work, the user must provide
        "shot_line", "gun", "shot_point", "cable", "channel". For improved
        common-channel performance consider modifying the chunksize to be
        (1, 1, 32, 1, 32, 2048) for good common-shot and common-channel
        performance or (1, 1, 128, 1, 1, 2048) for common-channel
        performance.

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
        as described above. The below configuration will yield 1MB chunks:

        >>> segy_to_mdio(
        ...     segy_path="prefix/cdp_offset_file.segy",
        ...     mdio_path_or_buffer="s3://bucket/cdp_offset_file.mdio",
        ...     index_bytes=(21,),
        ...     index_types=("int32",),
        ...     index_names=("cdp",),
        ...     chunksize=(4, 1024),
        ...     grid_overrides={"NonBinned": True, "chunksize": 64},
        ... )

        A more complicated case where you may have a 5D dataset that is not
        binned in Offset and Azimuth directions can be ingested like below.
        However, the Offset and Azimuth dimensions will be combined to "trace"
        dimension. The below configuration will yield 1MB chunks.

        >>> segy_to_mdio(
        ...     segy_path="prefix/cdp_offset_file.segy",
        ...     mdio_path_or_buffer="s3://bucket/cdp_offset_file.mdio",
        ...     index_bytes=(189, 193),
        ...     index_types=("int32", "int32"),
        ...     index_names=("inline", "crossline"),
        ...     chunksize=(4, 4, 1024),
        ...     grid_overrides={"NonBinned": True, "chunksize": 64},
        ... )

        For dataset with expected duplicate traces we have the following
        parameterization. This will use the same logic as NonBinned with
        a fixed chunksize of 1. The other keys are still important. The
        below example allows multiple traces per receiver (i.e. reshoot).

        >>> segy_to_mdio(
        ...     segy_path="prefix/cdp_offset_file.segy",
        ...     mdio_path_or_buffer="s3://bucket/cdp_offset_file.mdio",
        ...     index_bytes=(9, 213, 13),
        ...     index_types=("int32", "int16", "int32"),
        ...     index_names=("shot", "cable", "chan"),
        ...     chunksize=(8, 2, 256, 512),
        ...     grid_overrides={"HasDuplicates": True},
        ... )
    """
    num_index = len(index_bytes)

    if chunksize is not None:
        if len(chunksize) != len(index_bytes) + 1:
            message = (
                f"Length of chunks={len(chunksize)} must be ",
                f"equal to array dimensions={len(index_bytes) + 1}",
            )
            raise ValueError(message)

    if storage_options is None:
        storage_options = {}

    store = process_url(
        url=mdio_path_or_buffer,
        mode="w",
        storage_options=storage_options,
        memory_cache_size=0,  # Making sure disk caching is disabled,
        disk_cache=False,  # Making sure disk caching is disabled
    )

    # Read file specific metadata, build grid, and live trace mask.
    with segyio.open(
        filename=segy_path, mode="r", ignore_geometry=True, endian=endian
    ) as segy_handle:
        text_header = parse_text_header(segy_handle)
        binary_header = parse_binary_header(segy_handle)
        num_traces = segy_handle.tracecount

    index_types = parse_index_types(index_types, num_index)

    dimensions, chunksize, index_headers = get_grid_plan(
        segy_path=segy_path,
        segy_endian=endian,
        index_bytes=index_bytes,
        index_names=index_names,
        index_types=index_types,
        binary_header=binary_header,
        return_headers=True,
        chunksize=chunksize,
        grid_overrides=grid_overrides,
    )

    # Make grid and build global live trace mask
    grid = Grid(dims=dimensions)

    grid_density_qc(grid, num_traces)

    grid.build_map(index_headers)

    # Check grid validity by comparing trace numbers
    if np.sum(grid.live_mask) != num_traces:
        for dim_name in grid.dim_names:
            dim_min, dim_max = grid.get_min(dim_name), grid.get_max(dim_name)
            logger.warning(f"{dim_name} min: {dim_min} max: {dim_max}")
        logger.warning(f"Ingestion grid shape: {grid.shape}.")
        raise GridTraceCountError(np.sum(grid.live_mask), num_traces)

    zarr_root = create_zarr_hierarchy(
        store=store,
        overwrite=overwrite,
    )

    # Get UTC time, then add local timezone information offset.
    iso_datetime = datetime.now(timezone.utc).isoformat()

    write_attribute(name="created", zarr_group=zarr_root, attribute=iso_datetime)
    write_attribute(name="api_version", zarr_group=zarr_root, attribute=API_VERSION)

    dimensions_dict = [dim.to_dict() for dim in dimensions]
    write_attribute(name="dimension", zarr_group=zarr_root, attribute=dimensions_dict)

    # Write trace count
    trace_count = np.count_nonzero(grid.live_mask)
    write_attribute(name="trace_count", zarr_group=zarr_root, attribute=trace_count)

    # Note, live mask is not chunked since it's bool and small.
    zarr_root["metadata"].create_dataset(
        data=grid.live_mask,
        name="live_mask",
        shape=grid.shape[:-1],
        chunks=-1,
        dimension_separator="/",
    )

    write_attribute(
        name="text_header",
        zarr_group=zarr_root["metadata"],
        attribute=text_header,
    )

    write_attribute(
        name="binary_header",
        zarr_group=zarr_root["metadata"],
        attribute=binary_header,
    )

    if chunksize is None:
        dim_count = len(index_headers) + 1
        if dim_count == 2:
            chunksize = (512,) * 2

        elif dim_count == 3:
            chunksize = (64,) * 3

        else:
            msg = (
                f"Default chunking for {dim_count}-D seismic data is "
                "not implemented yet. Please explicity define chunk sizes."
            )
            raise NotImplementedError(msg)

        suffix = [str(x) for x in range(dim_count)]
        suffix = "".join(suffix)
    else:
        suffix = [dim_chunks if dim_chunks > 0 else None for dim_chunks in chunksize]
        suffix = [str(idx) for idx, value in enumerate(suffix) if value is not None]
        suffix = "".join(suffix)

    stats = blocked_io.to_zarr(
        segy_path=segy_path,
        segy_endian=endian,
        grid=grid,
        data_root=zarr_root["data"],
        metadata_root=zarr_root["metadata"],
        name="_".join(["chunked", suffix]),
        dtype="float32",
        chunks=chunksize,
        lossless=lossless,
        compression_tolerance=compression_tolerance,
    )

    for key, value in stats.items():
        write_attribute(name=key, zarr_group=zarr_root, attribute=value)

    # Non-cached store for consolidating metadata.
    # If caching is enabled the metadata may fall out of cache hence
    # creating an incomplete `.zmetadata` file.
    store_nocache = process_url(
        url=mdio_path_or_buffer,
        mode="r+",
        storage_options=storage_options,
        memory_cache_size=0,  # Making sure disk caching is disabled,
        disk_cache=False,  # Making sure disk caching is disabled
    )

    zarr.consolidate_metadata(store_nocache)
