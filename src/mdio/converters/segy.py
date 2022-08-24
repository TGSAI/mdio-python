"""Conversion from SEG-Y to MDIO."""


from __future__ import annotations

from datetime import datetime
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


API_VERSION = metadata.version("multidimio")
BACKENDS = ["s3", "gcs", "gs"]


def segy_to_mdio(
    segy_path: str,
    mdio_path_or_buffer: str,
    index_bytes: Sequence[int],
    index_names: Sequence[str] | None = None,
    index_lengths: Sequence[int] | None = None,
    chunksize: Sequence[int] | None = None,
    endian: str = "big",
    lossless: bool = True,
    compression_ratio: int | float = 4,
    storage_options: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Convert SEG-Y file to MDIO format.

    Args:
        segy_path: Path to the input SEG-Y file
        mdio_path_or_buffer: Output path for MDIO file
        index_bytes: Tuple of the byte location for the index attributes
        index_names: Tuple of the index names for the index attributes
        index_lengths: Tuple of the byte lengths for the index attributes
            Default is 4-byte for each index key.
        chunksize : Override default chunk size, which is (32, 32, 512)
        endian: Endianness of the input SEG-Y. Rev.2 allows little endian.
            Default is 'big'. Must be in `{"big", "little"}`
        lossless: Lossless Blosc with zstandard, or ZFP with fixed precision.
        compression_ratio: Approximate compression ratio for ZFP compression.
            Will be ignored if `lossless=True`
        storage_options: Storage options for the cloud storage backend.
            Default is `None` (will assume anonymous)
        overwrite: Toggle for overwriting existing store

    Raises:
        GridTraceCountError: Raised if grid won't hold all traces in the
            SEG-Y file.
        ValueError: If length of chunk sizes don't match number of dimensions.
    """
    num_index = len(index_bytes)

    # By default, we chunk non-sample directions by 32; and sample by 512.
    chunksize = (32,) * num_index + (512,) if chunksize is None else chunksize

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

    write_attribute(name="created", zarr_group=zarr_root, attribute=str(datetime.now()))
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
        compression_ratio=compression_ratio,
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
