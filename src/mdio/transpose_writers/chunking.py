"""Repartitioning operations for MDIO Variables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Hashable
import logging

import numpy as np
import dask
from dask import array as dask_array
from dask.diagnostics import ProgressBar
from xarray import DataArray

from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.api.io import _normalize_path
from mdio.builder.xarray_builder import _compressor_to_encoding


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
    new_variable: str,
    chunk_grid: RegularChunkGrid | RectilinearChunkGrid,
    compressor: ZFP | Blosc | None = None,
    copy_metadata: bool = True,
    num_workers: int = 4,
    scheduler: str = "processes",
    show_progress: bool = True,
) -> None:
    """Add a new Variable to the Dataset with different chunking and compression.

    Copies data from the source Variable to the new Variable to create a different access pattern.
    """

    # --- Validation (unchanged-ish) -----------------------------------------
    _validate_inputs(new_variable, chunk_grid, compressor)

    normed_path = _normalize_path(dataset_path)
    ds = open_mdio(normed_path)

    source_var = ds[source_variable]
    dims = source_var.dims

    # New chunk shape from the chunk grid
    new_chunks = tuple(chunk_grid.configuration.chunk_shape)

    if len(dims) != len(new_chunks):
        logger.warning(
            "Original variable %r has dimensions %r, but new chunk shape %r was provided. "
            "Behavior is currently undefined.",
            source_variable,
            dims,
            new_chunks,
        )

    original_chunks = source_var.encoding.get("chunks", None)

    if original_chunks is not None:
        for chunk in original_chunks:
            if chunk is None:
                logger.warning(
                    "Original chunk %r is None. Behavior is currently undefined.", chunk
                )

        original_chunk_size_mb = (
            np.prod(original_chunks) * source_var.dtype.itemsize / (1024**2)
        )
    else:
        original_chunk_size_mb = float("nan")

    new_chunk_size_mb = np.prod(new_chunks) * source_var.dtype.itemsize / (1024**2)

    logger.info(f"Original chunks: {original_chunks} (~{original_chunk_size_mb:.1f} MB)")
    logger.info(f"New chunks: {new_chunks} (~{new_chunk_size_mb:.1f} MB)")
    logger.info(
        "Estimated memory per worker: ~%.1f MB (includes rechunking overhead)",
        new_chunk_size_mb * 3,
    )

    # --- Build rechunked view of the source data ----------------------------
    # Construct a chunk mapping for xarray/dask
    chunk_mapping: dict[Hashable, int] = dict(zip(dims, new_chunks, strict=True))

    # Lazily rechunk the source variable to the *target* zarr chunks
    # This creates a dask graph that reads original chunks, assembles them
    # into new_chunks, and yields fresh ndarray blocks (already C-contiguous).
    source_rechunked = source_var.chunk(chunk_mapping)

    # --- Build the new variable DataArray -----------------------------------
    # Data is the rechunked dask array; coords/dims match the source.
    if copy_metadata:
        new_attrs = source_var.attrs.copy()
    else:
        new_attrs = {}

    new_da = DataArray(
        data=source_rechunked.data,
        dims=source_var.dims,
        coords=source_var.coords,
        attrs=new_attrs,
        name=new_variable,
    )

    # Wrap into a Dataset so we can write just this variable
    new_ds = new_da.to_dataset(name=new_variable)

    # --- Encoding (chunks + compressor) -------------------------------------
    if copy_metadata:
        new_encoding: dict[str, Any] = source_var.encoding.copy()
    else:
        new_encoding = {}

    # Ensure encoding chunks match the rechunked dask chunks
    new_encoding["chunks"] = new_chunks

    if compressor is not None:
        new_encoding.update(_compressor_to_encoding(compressor))

    new_ds[new_variable].encoding = new_encoding

    # --- Clean up attrs that conflict with consolidated metadata ------------
    _remove_fillvalue_attrs(new_ds)

    # Drop non-dimensional coords to avoid chunk conflicts
    coords_to_drop = [coord for coord in new_ds.coords if coord not in new_ds.dims]
    if coords_to_drop:
        new_ds = new_ds.drop_vars(coords_to_drop)

    # --- Configure Dask and write to MDIO -----------------------------------
    dask_config: dict[str, Any] = {"scheduler": scheduler}
    if scheduler in ("processes", "threads"):
        dask_config["num_workers"] = num_workers

    logger.info("Writing data with %d %s workers", num_workers, scheduler)

    with dask.config.set(**dask_config):
        if show_progress:
            with ProgressBar():
                to_mdio(new_ds, normed_path, mode="a", compute=True)
        else:
            to_mdio(new_ds, normed_path, mode="a", compute=True)

    logger.info("Variable copy complete: %r -> %r", source_variable, new_variable)

    
