"""Conversion from SEG-Y to MDIO."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc
from segy import SegyFile
from segy.config import SegySettings
from segy.schema import HeaderField

from mdio.converters.exceptions import EnvironmentFormatError
from mdio.converters.exceptions import GridTraceCountError
from mdio.converters.exceptions import GridTraceSparsityError
from mdio.core import Grid
from mdio.core.factory import MDIOCreateConfig
from mdio.core.factory import MDIOVariableConfig
from mdio.core.factory import create_empty
from mdio.core.utils_write import write_attribute
from mdio.segy import blocked_io
from mdio.segy.compat import mdio_segy_spec
from mdio.segy.utilities import get_grid_plan

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

try:
    import zfpy  # Base library
    from numcodecs import ZFPY  # Codec
except ImportError:
    ZFPY = None
    zfpy = None


logger = logging.getLogger(__name__)


def grid_density_qc(grid: Grid, num_traces: int) -> None:
    """Quality control for sensible grid density during SEG-Y to MDIO conversion.

    This function checks the density of the proposed grid by comparing the total possible traces
    (`grid_traces`) to the actual number of traces in the SEG-Y file (`num_traces`). A warning is
    logged if the sparsity ratio (`grid_traces / num_traces`) exceeds a configurable threshold,
    indicating potential inefficiency or misconfiguration.

    The warning threshold is set via the environment variable `MDIO__GRID__SPARSITY_RATIO_WARN`
    (default 2), and the error threshold via `MDIO__GRID__SPARSITY_RATIO_LIMIT` (default 10). To
    suppress the exception (but still log warnings), set `MDIO_IGNORE_CHECKS=1`.

    Args:
        grid: The Grid instance to check.
        num_traces: Expected number of traces from the SEG-Y file.

    Raises:
        GridTraceSparsityError: If the sparsity ratio exceeds `MDIO__GRID__SPARSITY_RATIO_LIMIT`
            and `MDIO_IGNORE_CHECKS` is not set to a truthy value (e.g., "1", "true").
        EnvironmentFormatError: If `MDIO__GRID__SPARSITY_RATIO_WARN` or
            `MDIO__GRID__SPARSITY_RATIO_LIMIT` cannot be converted to a float.
    """
    # Calculate total possible traces in the grid (excluding sample dimension)
    grid_traces = np.prod(grid.shape[:-1], dtype=np.uint64)

    # Handle division by zero if num_traces is 0
    sparsity_ratio = float("inf") if num_traces == 0 else grid_traces / num_traces

    # Fetch and validate environment variables
    warning_ratio_env = os.getenv("MDIO__GRID__SPARSITY_RATIO_WARN", "2")
    error_ratio_env = os.getenv("MDIO__GRID__SPARSITY_RATIO_LIMIT", "10")
    ignore_checks_env = os.getenv("MDIO_IGNORE_CHECKS", "false").lower()
    ignore_checks = ignore_checks_env in ("1", "true", "yes", "on")

    try:
        warning_ratio = float(warning_ratio_env)
    except ValueError as e:
        raise EnvironmentFormatError("MDIO__GRID__SPARSITY_RATIO_WARN", "float") from e  # noqa: EM101

    try:
        error_ratio = float(error_ratio_env)
    except ValueError as e:
        raise EnvironmentFormatError("MDIO__GRID__SPARSITY_RATIO_LIMIT", "float") from e  # noqa: EM101

    # Check sparsity
    should_warn = sparsity_ratio > warning_ratio
    should_error = sparsity_ratio > error_ratio and not ignore_checks

    # Early return if everything is OK
    # Prepare message for warning or error
    if not should_warn and not should_error:
        return

    # Build warning / error message
    dims = dict(zip(grid.dim_names, grid.shape, strict=True))
    msg = (
        f"Ingestion grid is sparse. Sparsity ratio: {sparsity_ratio:.2f}. "
        f"Ingestion grid: {dims}. "
        f"SEG-Y trace count: {num_traces}, grid trace count: {grid_traces}."
    )
    for dim_name in grid.dim_names:
        dim_min = grid.get_min(dim_name)
        dim_max = grid.get_max(dim_name)
        msg += f"\n{dim_name} min: {dim_min} max: {dim_max}"

    # Log warning if sparsity exceeds warning threshold
    if should_warn:
        logger.warning(msg)

    # Raise error if sparsity exceeds error threshold and checks are not ignored
    if should_error:
        raise GridTraceSparsityError(grid.shape, num_traces, msg)


def get_compressor(lossless: bool, compression_tolerance: float = -1) -> Blosc | ZFPY | None:
    """Get the appropriate compressor for the seismic traces."""
    if lossless:
        compressor = Blosc("zstd")
    else:
        if zfpy is None or ZFPY is None:
            msg = (
                "Lossy compression requires the 'zfpy' library. It is not installed in your "
                "environment. To proceed, please install 'zfpy' or install mdio `lossy` extra."
            )
            raise ImportError(msg)

        compressor = ZFPY(mode=zfpy.mode_fixed_accuracy, tolerance=compression_tolerance)
    return compressor


def segy_to_mdio(  # noqa: PLR0913, PLR0915, PLR0912
    segy_path: str | Path,
    mdio_path_or_buffer: str | Path,
    index_bytes: Sequence[int],
    index_names: Sequence[str] | None = None,
    index_types: Sequence[str] | None = None,
    chunksize: tuple[int, ...] | None = None,
    lossless: bool = True,
    compression_tolerance: float = 0.01,
    storage_options_input: dict[str, Any] | None = None,
    storage_options_output: dict[str, Any] | None = None,
    overwrite: bool = False,
    grid_overrides: dict | None = None,
) -> None:
    """Convert SEG-Y file to MDIO format.

    MDIO allows ingesting flattened seismic surveys in SEG-Y format into a multidimensional tensor
    that represents the correct geometry of the seismic dataset.

    The SEG-Y file must be on disk, MDIO currently does not support reading SEG-Y directly from
    the cloud object store.

    The output MDIO file can be local or on the cloud. For local files, a UNIX or Windows path is
    sufficient. However, for cloud stores, an appropriate protocol must be provided. See examples
    for more details.

    The SEG-Y headers for indexing must also be specified. The index byte locations (starts from 1)
    are the minimum amount of information needed to index the file. However, we suggest giving
    names to the index dimensions, and if needed providing the header lengths if they are not
    standard. By default, all header entries are assumed to be 4-byte long.

    The chunk size depends on the data type, however, it can be chosen to accommodate any
    workflow's access patterns. See examples below for some common use cases.

    By default, the data is ingested with LOSSLESS compression. This saves disk space in the range
    of 20% to 40%. MDIO also allows data to be compressed using the ZFP compressor's fixed rate
    lossy compression. If lossless parameter is set to False and MDIO was installed using the lossy
    extra; then the data will be compressed to approximately 30% of its original size and will be
    perceptually lossless. The compression ratio can be adjusted using the option compression_ratio
    (integer). Higher values will compress more, but will introduce artifacts.

    Args:
        segy_path: Path to the input SEG-Y file
        mdio_path_or_buffer: Output path for the MDIO file, either local or cloud-based (e.g.,
            with `s3://`, `gcs://`, or `abfs://` protocols).
        index_bytes: Tuple of the byte location for the index attributes
        index_names: List of names for the index dimensions. If not provided, defaults to `dim_0`,
            `dim_1`, ..., with the last dimension named `sample`.
        index_types: Tuple of the data-types for the index attributes. Must be in {"int16, int32,
            float16, float32, ibm32"}. Default is 4-byte integers for each index key.
        chunksize: Tuple specifying the chunk sizes for each dimension of the array. It must match
            the number of dimensions in the input array.
        lossless: If True, uses lossless Blosc compression with zstandard. If False, uses ZFP lossy
            compression (requires `zfpy` library).
        compression_tolerance: Tolerance for ZFP compression in lossy mode. Ignored if
            `lossless=True`. Default is 0.01, providing ~70% size reduction.
        storage_options_input: Dictionary of storage options for the SEGY input output file (e.g.,
            cloud credentials). Defaults to None.
        storage_options_output: Dictionary of storage options for the MDIO output output file
            (e.g., cloud credentials). Defaults to None.
        overwrite: If True, overwrites existing MDIO file at the specified path.
        grid_overrides: Option to add grid overrides. See examples.

    Raises:
        GridTraceCountError: Raised if grid won't hold all traces in the SEG-Y file.
        ValueError: If length of chunk sizes don't match number of dimensions.
        NotImplementedError: If can't determine chunking automatically for 4D+.

    Examples:
        If we are working locally and ingesting a 3D post-stack seismic file, we can use the
        following example. This will ingest with default chunks of 128 x 128 x 128.

        >>> from mdio import segy_to_mdio
        >>>
        >>>
        >>> segy_to_mdio(
        ...     segy_path="prefix1/file.segy",
        ...     mdio_path_or_buffer="prefix2/file.mdio",
        ...     index_bytes=(189, 193),
        ...     index_names=("inline", "crossline")
        ... )

        If we are on Amazon Web Services, we can do it like below. The protocol before the URL can
        be `s3` for AWS, `gcs` for Google Cloud, and `abfs` for Microsoft Azure. In this example we
        also change the chunk size as a demonstration.

        >>> segy_to_mdio(
        ...     segy_path="prefix/file.segy",
        ...     mdio_path_or_buffer="s3://bucket/file.mdio",
        ...     index_bytes=(189, 193),
        ...     index_names=("inline", "crossline"),
        ...     chunksize=(64, 64, 512),
        ... )

        Another example of loading a 4D seismic such as 3D seismic pre-stack gathers is below. This
        will allow us to extract offset planes efficiently or run things in a local neighborhood
        very efficiently.

        >>> segy_to_mdio(
        ...     segy_path="prefix/file.segy",
        ...     mdio_path_or_buffer="s3://bucket/file.mdio",
        ...     index_bytes=(189, 193, 37),
        ...     index_names=("inline", "crossline", "offset"),
        ...     chunksize=(16, 16, 16, 512),
        ... )

        We can override the dataset grid by the `grid_overrides` parameter. This allows us to
        ingest files that don't conform to the true geometry of the seismic acquisition.

        For example if we are ingesting 3D seismic shots that don't have a cable number and channel
        numbers are sequential (i.e. each cable doesn't start with channel number 1; we can tell
        MDIO to ingest this with the correct geometry by calculating cable numbers and wrapped
        channel numbers. Note the missing byte location and word length for the "cable" index.

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

        If we do have cable numbers in the headers, but channels are still sequential (aka.
        unwrapped), we can still ingest it like this.

        >>> segy_to_mdio(
        ...     segy_path="prefix/shot_file.segy",
        ...     mdio_path_or_buffer="s3://bucket/shot_file.mdio",
        ...     index_bytes=(17, 137, 13),
        ...     index_lengths=(4, 2, 4),
        ...     index_names=("shot_point", "cable", "channel"),
        ...     chunksize=(8, 2, 128, 1024),
        ...     grid_overrides={"ChannelWrap": True, "ChannelsPerCable": 800},
        ... )

        For shot gathers with channel numbers and wrapped channels, no grid overrides necessary.

        In cases where the user does not know if the input has unwrapped channels but desires to
        store with wrapped channel index use:
        >>> grid_overrides = {
        ...    "AutoChannelWrap": True,
        ...    "AutoChannelTraceQC": 1000000
        ... }

        For ingestion of pre-stack streamer data where the user needs to access/index
        *common-channel gathers* (single gun) then the following strategy can be used to densely
        ingest while indexing on gun number:

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

        For AutoShotWrap and AutoChannelWrap to work, the user must provide "shot_line", "gun",
        "shot_point", "cable", "channel". For improved common-channel performance consider
        modifying the chunksize to be (1, 1, 32, 1, 32, 2048) for good common-shot and
        common-channel performance or (1, 1, 128, 1, 1, 2048) for common-channel performance.

        For cases with no well-defined trace header for indexing a NonBinned grid override is
        provided.This creates the index and attributes an incrementing integer to the trace for
        the index based on first in first out. For example a CDP and Offset keyed file might have a
        header for offset as real world offset which would result in a very sparse populated index.
        Instead, the following override will create a new index from 1 to N, where N is the number
        of offsets within a CDP ensemble. The index to be auto generated is called "trace". Note
        the required "chunksize" parameter in the grid override. This is due to the non-binned
        ensemble chunksize is irrelevant to the index dimension chunksizes and has to be specified
        in the grid override itself. Note the lack of offset, only indexing CDP, providing CDP
        header type, and chunksize for only CDP and Sample dimension. The chunksize for non-binned
        dimension is in the grid overrides as described above. The below configuration will yield
        1MB chunks:

        >>> segy_to_mdio(
        ...     segy_path="prefix/cdp_offset_file.segy",
        ...     mdio_path_or_buffer="s3://bucket/cdp_offset_file.mdio",
        ...     index_bytes=(21,),
        ...     index_types=("int32",),
        ...     index_names=("cdp",),
        ...     chunksize=(4, 1024),
        ...     grid_overrides={"NonBinned": True, "chunksize": 64},
        ... )

        A more complicated case where you may have a 5D dataset that is not binned in Offset and
        Azimuth directions can be ingested like below. However, the Offset and Azimuth dimensions
        will be combined to "trace" dimension. The below configuration will yield 1MB chunks.

        >>> segy_to_mdio(
        ...     segy_path="prefix/cdp_offset_file.segy",
        ...     mdio_path_or_buffer="s3://bucket/cdp_offset_file.mdio",
        ...     index_bytes=(189, 193),
        ...     index_types=("int32", "int32"),
        ...     index_names=("inline", "crossline"),
        ...     chunksize=(4, 4, 1024),
        ...     grid_overrides={"NonBinned": True, "chunksize": 64},
        ... )

        For dataset with expected duplicate traces we have the following parameterization. This
        will use the same logic as NonBinned with a fixed chunksize of 1. The other keys are still
        important. The below example allows multiple traces per receiver (i.e. reshoot).

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
    index_names = index_names or [f"dim_{i}" for i in range(len(index_bytes))]
    index_types = index_types or ["int32"] * len(index_bytes)

    if chunksize is not None and len(chunksize) != len(index_bytes) + 1:
        message = (
            f"Length of chunks={len(chunksize)} must be equal to array "
            f"dimensions={len(index_bytes) + 1}"
        )
        raise ValueError(message)

    # Handle storage options and check permissions etc
    storage_options_input = storage_options_input or {}
    storage_options_output = storage_options_output or {}

    # Open SEG-Y with MDIO's SegySpec. Endianness will be inferred.
    mdio_spec = mdio_segy_spec()
    segy_settings = SegySettings(storage_options=storage_options_input)
    segy = SegyFile(url=segy_path, spec=mdio_spec, settings=segy_settings)

    text_header = segy.text_header
    binary_header = segy.binary_header
    num_traces = segy.num_traces

    # Index the dataset using a spec that interprets the user provided index headers.
    index_fields = []
    for name, byte, format_ in zip(index_names, index_bytes, index_types, strict=True):
        index_fields.append(HeaderField(name=name, byte=byte, format=format_))
    mdio_spec_grid = mdio_spec.customize(trace_header_fields=index_fields)
    segy_grid = SegyFile(url=segy_path, spec=mdio_spec_grid, settings=segy_settings)

    dimensions, chunksize, index_headers = get_grid_plan(
        segy_file=segy_grid,
        return_headers=True,
        chunksize=chunksize,
        grid_overrides=grid_overrides,
    )
    grid = Grid(dims=dimensions)
    grid_density_qc(grid, num_traces)
    grid.build_map(index_headers)

    # Check grid validity by ensuring every trace's header-index is within dimension bounds
    valid_mask = np.ones(grid.num_traces, dtype=bool)
    for d_idx in range(len(grid.header_index_arrays)):
        coords = grid.header_index_arrays[d_idx]
        valid_mask &= coords < grid.shape[d_idx]
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count != num_traces:
        for dim_name in grid.dim_names:
            dim_min = grid.get_min(dim_name)
            dim_max = grid.get_max(dim_name)
            logger.warning("%s min: %s max: %s", dim_name, dim_min, dim_max)
        logger.warning("Ingestion grid shape: %s.", grid.shape)
        raise GridTraceCountError(valid_count, num_traces)

    import gc

    del valid_mask
    gc.collect()

    if chunksize is None:
        dim_count = len(index_names) + 1
        if dim_count == 2:  # noqa: PLR2004
            chunksize = (512,) * 2

        elif dim_count == 3:  # noqa: PLR2004
            chunksize = (64,) * 3

        else:
            msg = (
                f"Default chunking for {dim_count}-D seismic data is not implemented yet. "
                "Please explicity define chunk sizes."
            )
            raise NotImplementedError(msg)

        suffix = [str(x) for x in range(dim_count)]
        suffix = "".join(suffix)
    else:
        suffix = [dim_chunks if dim_chunks > 0 else None for dim_chunks in chunksize]
        suffix = [str(idx) for idx, value in enumerate(suffix) if value is not None]
        suffix = "".join(suffix)

    compressors = get_compressor(lossless, compression_tolerance)
    header_dtype = segy.spec.trace.header.dtype.newbyteorder("=")
    var_conf = MDIOVariableConfig(
        name=f"chunked_{suffix}",
        dtype="float32",
        chunks=chunksize,
        compressors=compressors,
        header_dtype=header_dtype,
    )
    config = MDIOCreateConfig(path=mdio_path_or_buffer, grid=grid, variables=[var_conf])

    root_group = create_empty(
        config,
        overwrite=overwrite,
        storage_options=storage_options_output,
        consolidate_meta=False,
    )
    data_group = root_group["data"]
    meta_group = root_group["metadata"]
    data_array = data_group[f"chunked_{suffix}"]
    header_array = meta_group[f"chunked_{suffix}_trace_headers"]

    live_mask_array = meta_group["live_mask"]
    # 'live_mask_array' has the same first N–1 dims as 'grid.shape[:-1]'
    # Build a ChunkIterator over the live_mask (no sample axis)
    from mdio.core.indexing import ChunkIterator

    chunker = ChunkIterator(live_mask_array, chunk_samples=True)
    for chunk_indices in chunker:
        # chunk_indices is a tuple of N–1 slice objects
        trace_ids = grid.get_traces_for_chunk(chunk_indices)
        if trace_ids.size == 0:
            # Free memory immediately for empty chunks
            del trace_ids
            continue

        # Build a temporary boolean block of shape = chunk shape
        block = np.zeros(tuple(sl.stop - sl.start for sl in chunk_indices), dtype=bool)

        # Compute local coords within this block for each trace_id
        local_coords: list[np.ndarray] = []
        for dim_idx, sl in enumerate(chunk_indices):
            hdr_arr = grid.header_index_arrays[dim_idx]
            # Optimize memory usage: hdr_arr and trace_ids are already uint32,
            # sl.start is int, so result should naturally be int32/uint32.
            # Avoid unnecessary astype conversion to int64.
            indexed_coords = hdr_arr[trace_ids]  # uint32 array
            local_idx = indexed_coords - sl.start  # remains uint32
            # Free indexed_coords immediately
            del indexed_coords

            # Only convert dtype if necessary for indexing (numpy requires int for indexing)
            if local_idx.dtype != np.intp:
                local_idx = local_idx.astype(np.intp)
            local_coords.append(local_idx)
            # local_idx is now owned by local_coords list, safe to continue

        # Free trace_ids as soon as we're done with it
        del trace_ids

        # Mark live cells in the temporary block
        block[tuple(local_coords)] = True

        # Free local_coords immediately after use
        del local_coords

        # Write the entire block to Zarr at once
        live_mask_array.set_basic_selection(selection=chunk_indices, value=block)

        # Free block immediately after writing
        del block

        # Force garbage collection periodically to free memory aggressively
        gc.collect()

    # Final cleanup
    del live_mask_array
    del chunker
    gc.collect()

    nonzero_count = grid.num_traces

    write_attribute(name="trace_count", zarr_group=root_group, attribute=nonzero_count)
    write_attribute(name="text_header", zarr_group=meta_group, attribute=text_header.split("\n"))
    write_attribute(name="binary_header", zarr_group=meta_group, attribute=binary_header.to_dict())

    # Write traces
    stats = blocked_io.to_zarr(
        segy_file=segy,
        grid=grid,
        data_array=data_array,
        header_array=header_array,
    )

    # Write actual stats
    for key, value in stats.items():
        write_attribute(name=key, zarr_group=root_group, attribute=value)

    zarr.consolidate_metadata(root_group.store)
