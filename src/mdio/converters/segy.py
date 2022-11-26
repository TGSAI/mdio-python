"""Conversion from SEG-Y to MDIO."""


from __future__ import annotations

from datetime import datetime
from datetime import timezone
from importlib import metadata
from typing import Any
from typing import Sequence

import numpy as np
import segyio
import zarr

from mdio.api.io_utils import process_url
from mdio.converters.exceptions import GridTraceCountError
from mdio.core import Grid
from mdio.core.utils_write import write_attribute
from mdio.segy import blocked_io
from mdio.segy.helpers_segy import create_zarr_hierarchy
from mdio.segy.parsers import parse_binary_header
from mdio.segy.parsers import parse_text_header
from mdio.segy.utilities import get_grid_plan


try:
    API_VERSION = metadata.version("multidimio")
except metadata.PackageNotFoundError:
    API_VERSION = "unknown"

BACKENDS = ["s3", "gcs", "gs", "az", "abfs"]


def segy_to_mdio(
    segy_path: str,
    mdio_path_or_buffer: str,
    index_bytes: Sequence[int],
    index_names: Sequence[str] | None = None,
    index_lengths: Sequence[int] | None = None,
    chunksize: Sequence[int] | None = None,
    endian: str = "big",
    lossless: bool = True,
    compression_tolerance: float = 0.01,
    storage_options: dict[str, Any] | None = None,
    overwrite: bool = False,
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
        index_lengths: Tuple of the byte lengths for the index attributes
            Default is 4-byte for each index key.
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
    """
    num_index = len(index_bytes)

    if chunksize is None:
        if num_index == 1:
            chunksize = (512,) * 2

        elif num_index == 2:
            chunksize = (64,) * 3

        else:
            msg = (
                f"Default chunking for {num_index + 1}-D seismic data is "
                "not implemented yet. Please explicity define chunk sizes."
            )
            raise NotImplementedError(msg)

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

    dimensions, index_headers = get_grid_plan(
        segy_path=segy_path,
        segy_endian=endian,
        index_bytes=index_bytes,
        index_names=index_names,
        index_lengths=index_lengths,
        binary_header=binary_header,
        return_headers=True,
    )

    # Make grid and build global live trace mask
    grid = Grid(dims=dimensions)
    grid.build_map(index_headers)

    # Check grid validity by comparing trace numbers
    if np.sum(grid.live_mask) != num_traces:
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
        suffix = [str(x) for x in range(len(index_bytes) + 1)]
        suffix = "".join(suffix)

    else:
        if len(chunksize) != len(index_bytes) + 1:
            message = (
                f"Length of chunks={len(chunksize)} must be ",
                f"equal to array dimensions={len(index_bytes) + 1}",
            )
            raise ValueError(message)

        suffix = [
            dim_chunksize if dim_chunksize > 0 else None for dim_chunksize in chunksize
        ]
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
