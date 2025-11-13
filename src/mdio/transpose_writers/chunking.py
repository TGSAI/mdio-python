"""Repartitioning operations for MDIO Variables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Any, Hashable
import logging
from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.api.io import _normalize_path
from mdio.builder.xarray_builder import _compressor_to_encoding
from dask.array import zeros_like
from xarray import DataArray as da


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from upath import UPath

    from mdio.builder.schemas.compressors import ZFP
    from mdio.builder.schemas.compressors import Blosc
    from mdio.builder.schemas.chunk_grid import RegularChunkGrid
    from mdio.builder.schemas.chunk_grid import RectilinearChunkGrid

def _normalize_chunks(
    original_chunks: tuple[int, ...] | None,
    new_chunks: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if original_chunks is None:
        return new_chunks

    return tuple(max(a, b) for a, b in zip(original_chunks, new_chunks, strict=True))
    

def _validate_inputs(
    new_variable: str | list[str],
    chunk_grid: RegularChunkGrid | RectilinearChunkGrid | list[RegularChunkGrid | RectilinearChunkGrid],
    compressor: ZFP | Blosc | list[ZFP | Blosc] | None,
) -> None:
    if isinstance(chunk_grid, list):
        if len(new_variable) != len(chunk_grid.chunk_shape):
            raise ValueError("new_variable and chunk_grid must have the same length")
    # if compressor is not None and len(compressor) != len(chunk_grid) or len(compressor) != 1:
    #     raise ValueError("chunk_grid and compressor must have the same length or be a single compressor")

    # TODO (BrianMichell): Remove task scoping validation
    # #0000
    if isinstance(chunk_grid, list):
        raise NotImplementedError("List of chunk grids is not supported yet")
    
    if isinstance(compressor, list):
        if compressor is not None and len(compressor) != len(chunk_grid) or len(compressor) != 1:
            raise ValueError("chunk_grid and compressor must have the same length or be a single compressor")
        raise NotImplementedError("List of compressors is not supported yet")


def from_variable(
    dataset_path: UPath | Path | str,
    source_variable: str,
    new_variable: str | list[str],
    chunk_grid: RegularChunkGrid | RectilinearChunkGrid | list[RegularChunkGrid | RectilinearChunkGrid],
    compressor: ZFP | Blosc | list[ZFP | Blosc] | None = None,
    copy_metadata: bool = True,
) -> None:
    """Add new Variable(s) to the Dataset with different chunking and compression.

    Copies data from the source Variable to the new Variable(s) to create different access patterns.

    Args:
        dataset_path: The path to a pre-existing MDIO Dataset.
        source_variable: The name of the existing Variable to copy data from.
        new_variable: The name(s) of the new Variable(s) to create.
        chunk_grid: The chunk grid to use for the new Variable(s). Length must match the number of new variables.
        compressor: The compressor to use for the new Variable(s). Length must match the number of new variables or be a single compressor.
        copy_metadata: Whether to copy the metadata from the source Variable to the new Variable(s).
    """
    _validate_inputs(new_variable, chunk_grid, compressor)

    normed_path = _normalize_path(dataset_path)
    original_ds = open_mdio(normed_path)

    dims = original_ds[source_variable].dims

    original_chunks = original_ds[source_variable].encoding.get("chunks", None)

    # TODO: This needs to be looped over for full function support
    new_chunks = chunk_grid.configuration.chunk_shape

    if len(dims) != len(new_chunks):
        logger.warning(f"Original variable {source_variable} has dimensions {dims}, but new chunk shape {new_chunks} were provided. Undefined behavior for now.")

    for chunk in original_chunks:
        if chunk is None:
            logger.warning(f"Original chunk {chunk} is None. Undefined behavior for now.")

    normalized_chunks = _normalize_chunks(original_chunks, new_chunks)

    print(f"Original chunks: {original_chunks}")
    print(f"New chunks: {new_chunks}")
    print(f"Normalized chunks: {normalized_chunks}")

    reopen_chunks = dict(zip(dims, normalized_chunks))

    # Step 1: Create new variable with lazy array
    source_var = original_ds[source_variable]
    lazy_array = zeros_like(source_var, chunks=new_chunks)
    
    # Step 2: Create new variable with specified encoding
    original_ds[new_variable] = da(lazy_array, dims=source_var.dims)
    
    if copy_metadata:
        original_ds[new_variable].attrs = source_var.attrs.copy()
    
    # Prepare encoding
    original_ds[new_variable].encoding = source_var.encoding.copy() if copy_metadata else {}
    original_ds[new_variable].encoding["chunks"] = new_chunks
    
    if compressor is not None:
        original_ds[new_variable].encoding.update(_compressor_to_encoding(compressor))
    
    # Step 3: Remove _FillValue from attrs to avoid conflicts
    for var_name in list(original_ds) + list(original_ds.coords):
        if "_FillValue" in original_ds[var_name].attrs:
            del original_ds[var_name].attrs["_FillValue"]
    
    # Step 4: Write new variable structure (metadata only)
    to_mdio(original_ds, normed_path, mode="a", compute=False)
    
    # Step 5: Reopen dataset for data copy
    ds_reopen = open_mdio(normed_path, chunks=reopen_chunks)
    
    # Remove _FillValue from attrs in reopened dataset to avoid conflicts
    for var_name in list(ds_reopen) + list(ds_reopen.coords):
        if "_FillValue" in ds_reopen[var_name].attrs:
            del ds_reopen[var_name].attrs["_FillValue"]
    
    # Step 6: Align chunks and lazy assignment from source to destination
    src = ds_reopen[source_variable]
    dst = ds_reopen[new_variable]
    
    # Rechunk source to match destination chunks for aligned copy
    src_rechunked = src.chunk(dst.chunks)
    ds_reopen[new_variable][:] = src_rechunked
    
    # Step 7: Compute and write the data (only the new variable)
    # Select only the new variable, drop coordinates to avoid chunk conflicts
    write_ds = ds_reopen[[new_variable]]
    coords_to_drop = [coord for coord in write_ds.coords if coord not in write_ds.dims]
    if coords_to_drop:
        write_ds = write_ds.drop_vars(coords_to_drop)
    to_mdio(write_ds, normed_path, mode="a", compute=True)
    
    logger.info("Variable copy complete")
    
