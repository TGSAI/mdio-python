"""Repartitioning operations for MDIO Variables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Any, Hashable
import logging
from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.api.io import _normalize_path
from mdio.builder.xarray_builder import _compressor_to_encoding
from dask import array as dask_array
from xarray import DataArray


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


def _remove_fillvalue_attrs(dataset: Any) -> None:
    """Remove _FillValue from all variable attrs to avoid conflicts with consolidated metadata."""
    # This is only relevant for Zarr v2 format.
    for var_name in list(dataset) + list(dataset.coords):
        if "_FillValue" in dataset[var_name].attrs:
            del dataset[var_name].attrs["_FillValue"]
    

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
    ds = open_mdio(normed_path)

    dims = ds[source_variable].dims
    original_chunks = ds[source_variable].encoding.get("chunks", None)

    # TODO: This needs to be looped over for full function support
    new_chunks = chunk_grid.configuration.chunk_shape

    if len(dims) != len(new_chunks):
        logger.warning(f"Original variable {source_variable} has dimensions {dims}, but new chunk shape {new_chunks} were provided. Undefined behavior for now.")

    for chunk in original_chunks:
        if chunk is None:
            logger.warning(f"Original chunk {chunk} is None. Undefined behavior for now.")

    logger.debug(f"Original chunks: {original_chunks}, New chunks: {new_chunks}")

    # Create new variable with lazy dask array (no data materialization)
    source_var = ds[source_variable]
    lazy_array = dask_array.empty(
        shape=source_var.shape,
        dtype=source_var.dtype,
        chunks=new_chunks,
    )
    ds[new_variable] = DataArray(lazy_array, dims=source_var.dims)
    
    if copy_metadata:
        ds[new_variable].attrs = source_var.attrs.copy()
    
    # Set up encoding with new chunks and optional compressor
    ds[new_variable].encoding = source_var.encoding.copy() if copy_metadata else {}
    ds[new_variable].encoding["chunks"] = new_chunks
    if compressor is not None:
        ds[new_variable].encoding.update(_compressor_to_encoding(compressor))
    
    # Write new variable structure (metadata only, don't include source variable)
    _remove_fillvalue_attrs(ds)
    write_metadata_ds = ds[[new_variable]]
    coords_to_drop = [coord for coord in write_metadata_ds.coords if coord not in write_metadata_ds.dims]
    if coords_to_drop:
        write_metadata_ds = write_metadata_ds.drop_vars(coords_to_drop)
    to_mdio(write_metadata_ds, normed_path, mode="a", compute=False)
    
    # Reopen dataset with optimized chunking for the copy operation
    normalized_chunks = _normalize_chunks(original_chunks, new_chunks)
    optimal_chunks = dict(zip(dims, normalized_chunks, strict=True))
    ds_copy = open_mdio(normed_path, chunks=optimal_chunks)
    
    # Perform lazy rechunked copy from source to destination
    src_rechunked = ds_copy[source_variable].chunk(ds_copy[new_variable].chunks)
    ds_copy[new_variable][:] = src_rechunked
    
    # Write only the new variable data (drop non-dimensional coordinates to avoid chunk conflicts)
    _remove_fillvalue_attrs(ds_copy)
    write_ds = ds_copy[[new_variable]]
    coords_to_drop = [coord for coord in write_ds.coords if coord not in write_ds.dims]
    if coords_to_drop:
        write_ds = write_ds.drop_vars(coords_to_drop)
    
    to_mdio(write_ds, normed_path, mode="a", compute=True)
    # TODO: I don't want to have to explicitly use Dask like this. It would be nice to just configure the scheduler and let to_mdio handle it on the backend
    # import dask
    # from dask.diagnostics import ProgressBar
    # with dask.config.set(scheduler="processes", num_workers=16):
    #     with ProgressBar():
    #         write_ds[[new_variable]].to_zarr(
    #             normed_path,
    #             mode="a",
    #             compute=True,
    #         )
    
    logger.info(f"Variable copy complete: '{source_variable}' -> '{new_variable}'")
    
