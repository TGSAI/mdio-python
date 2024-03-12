"""Convenience APIs for working with MDIO files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import zarr
from tqdm.auto import tqdm
from zarr import Blosc

from mdio.api.io_utils import process_url
from mdio.core.indexing import ChunkIterator


if TYPE_CHECKING:
    from numcodecs.abc import Codec
    from numpy.typing import NDArray
    from zarr import Array

    from mdio import MDIOAccessor
    from mdio import MDIOReader


def copy_mdio(  # noqa: PLR0913
    source: MDIOReader,
    dest_path_or_buffer: str,
    excludes: str = "",
    includes: str = "",
    storage_options: dict | None = None,
    overwrite: bool = False,
) -> None:
    """Copy MDIO file.

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
    if storage_options is None:
        storage_options = {}

    dest_store = process_url(
        url=dest_path_or_buffer,
        mode="w",
        storage_options=storage_options,
        memory_cache_size=0,
        disk_cache=False,
    )

    if_exists = "replace" if overwrite is True else "raise"

    zarr.copy_store(
        source=source.store,
        dest=dest_store,
        excludes=excludes,
        includes=includes,
        if_exists=if_exists,
    )

    if len(excludes) > 0:
        data_path = f"data/{excludes}"
        source_array = source.root[data_path]
        dimension_separator = source_array._dimension_separator

        zarr.zeros_like(
            source_array,
            store=dest_store,
            path=data_path,
            overwrite=overwrite,
            dimension_separator=dimension_separator,
        )


CREATE_KW = {
    "dimension_separator": "/",
    "write_empty_chunks": False,
}
MAX_BUFFER = 512


def create_rechunk_plan(
    source: MDIOAccessor,
    chunks_list: list[tuple[int, ...]],
    suffix_list: list[str],
    compressor: Codec | None = None,
    overwrite: bool = False,
) -> tuple[[list[Array]], list[Array], NDArray, ChunkIterator]:
    """Create rechunk plan based on source and user input.

    It will buffer 512 x n-dimensions in memory. Approximately
    128MB. However, if you need to adjust the buffer size, change
    the `MAX_BUFFER` variable in this module.

    Args:
        source: MDIO accessor instance. Data will be copied from here.
        chunks_list: List of tuples containing new chunk sizes.
        suffix_list: List of suffixes to append to new chunk sizes.
        compressor: Data compressor to use, optional. Default is Blosc('zstd').
        overwrite: Overwrite destination or not.

    Returns:
        Tuple containing the rechunk plan variables and iterator.

    Raises:
        NameError: if trying to write to original data.
    """
    data_group = source._data_group
    metadata_group = source._metadata_group

    data_array = source._traces
    metadata_array = source._headers
    live_mask = source.live_mask[:]

    metadata_arrs = []
    data_arrs = []

    header_compressor = Blosc("zstd")
    trace_compressor = Blosc("zstd") if compressor is None else compressor

    for chunks, suffix in zip(chunks_list, suffix_list):  # noqa: B905
        norm_chunks = [
            min(chunk, size) for chunk, size in zip(chunks, source.shape)  # noqa: B905
        ]

        if suffix == source.access_pattern:
            msg = f"Can't write over source data with suffix {suffix}"
            raise NameError(msg)

        metadata_arrs.append(
            metadata_group.zeros_like(
                name=f"chunked_{suffix}_trace_headers",
                data=metadata_array,
                chunks=norm_chunks[:-1],
                compressor=header_compressor,
                overwrite=overwrite,
                **CREATE_KW,
            )
        )

        data_arrs.append(
            data_group.zeros_like(
                name=f"chunked_{suffix}",
                data=data_array,
                chunks=norm_chunks,
                compressor=trace_compressor,
                overwrite=overwrite,
                **CREATE_KW,
            )
        )

    n_dimension = len(data_array.shape)
    dummy_array = zarr.empty_like(data_array, chunks=(MAX_BUFFER,) * n_dimension)
    iterator = ChunkIterator(dummy_array)

    return metadata_arrs, data_arrs, live_mask, iterator


def write_rechunked_values(  # noqa: PLR0913
    source: MDIOAccessor,
    suffix_list: list[str],
    metadata_arrs_out: list[Array],
    data_arrs_out: list[Array],
    live_mask: NDArray,
    iterator: ChunkIterator,
) -> None:
    """Create rechunk plan based on source and user input.

    Args:
        source: MDIO accessor instance. Data will be copied from here.
        suffix_list: List of suffixes to append to new chunk sizes.
        metadata_arrs_out: List of new metadata Zarr arrays.
        data_arrs_out: List of new data Zarr arrays.
        live_mask: Live mask to apply during copies.
        iterator: The chunk iterator to use.
    """
    suffix_names = ",".join(suffix_list)
    for slice_ in tqdm(iterator, desc=f"Rechunking to {suffix_names}", unit="chunk"):
        meta_slice = slice_[:-1]

        if live_mask[meta_slice].sum() == 0:
            continue

        for array in metadata_arrs_out:
            array[meta_slice] = source._headers[meta_slice]

        for array in data_arrs_out:
            array[slice_] = source._traces[slice_]

        zarr.consolidate_metadata(source.store)


def rechunk_batch(
    source: MDIOAccessor,
    chunks_list: list[tuple[int, ...]],
    suffix_list: list[str],
    compressor: Codec | None = None,
    overwrite: bool = False,
) -> None:
    """Rechunk MDIO file to multiple variables, reading it once.

    Args:
        source: MDIO accessor instance. Data will be copied from here.
        chunks_list: List of tuples containing new chunk sizes.
        suffix_list: List of suffixes to append to new chunk sizes.
        compressor: Data compressor to use, optional. Default is Blosc('zstd').
        overwrite: Overwrite destination or not.

    Examples:
        To rechunk multiple variables we can do things like:

        >>> accessor = MDIOAccessor(...)
        >>> rechunk_batch(
        >>>     accessor,
        >>>     chunks_list=[(1, 1024, 1024), (1024, 1, 1024), (1024, 1024, 1)],
        >>>     suffix_list=["fast_il", "fast_xl", "fast_z"],
        >>> )
    """
    plan = create_rechunk_plan(
        source,
        chunks_list=chunks_list,
        suffix_list=suffix_list,
        compressor=compressor,
        overwrite=overwrite,
    )

    write_rechunked_values(source, suffix_list, *plan)


def rechunk(
    source: MDIOAccessor,
    chunks: tuple[int, ...],
    suffix: str,
    compressor: Codec | None = None,
    overwrite: bool = False,
) -> None:
    """Rechunk MDIO file adding a new variable.

    Args:
        source: MDIO accessor instance. Data will be copied from here.
        chunks: Tuple containing chunk sizes for new rechunked array.
        suffix: Suffix to append to new rechunked array.
        compressor: Data compressor to use, optional. Default is Blosc('zstd').
        overwrite: Overwrite destination or not.

    Examples:
        To rechunk a single variable we can do this

        >>> accessor = MDIOAccessor(...)
        >>> rechunk(accessor, (1, 1024, 1024), suffix="fast_il")
    """
    # TODO(Anyone): Write tests for rechunking functions
    # https://github.com/TGSAI/mdio-python/issues/369
    rechunk_batch(source, [chunks], [suffix], compressor, overwrite)
