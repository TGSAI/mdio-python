"""Repartitioning operations for MDIO Variables."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING
import logging

import numpy as np
import dask
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback
from xarray import DataArray

from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.api.io import _normalize_path
from mdio.builder.xarray_builder import _compressor_to_encoding
from mdio.core.config import MDIOSettings


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from upath import UPath

    from mdio.builder.schemas.compressors import ZFP
    from mdio.builder.schemas.compressors import Blosc
    from mdio.builder.schemas.chunk_grid import RegularChunkGrid
    from mdio.builder.schemas.chunk_grid import RectilinearChunkGrid


def _remove_fillvalue_attrs(dataset: Any) -> None:
    """Remove _FillValue from all variable attrs to avoid conflicts with consolidated metadata.

    This is only relevant for Zarr v2 format.
    """
    for var_name in list(dataset.data_vars) + list(dataset.coords):
        if "_FillValue" in dataset[var_name].attrs:
            del dataset[var_name].attrs["_FillValue"]


def _validate_inputs(
    new_variable: str | list[str],
    chunk_grid: "RegularChunkGrid"
    | "RectilinearChunkGrid"
    | list["RegularChunkGrid" | "RectilinearChunkGrid"],
    compressor: "ZFP" | "Blosc" | list["ZFP" | "Blosc"] | None,
) -> None:
    """Validate basic shapes/types (no broadcasting here)."""
    logger.debug("Validating inputs: new_variable=%r, chunk_grid type=%s, compressor type=%s",
                 new_variable, type(chunk_grid), type(compressor))

    # new_variable must be str or non-empty list[str]
    if isinstance(new_variable, str):
        logger.debug("new_variable is a single string: %r", new_variable)
        pass
    elif isinstance(new_variable, list):
        if not new_variable:
            raise ValueError("new_variable list must not be empty")
        if not all(isinstance(v, str) for v in new_variable):
            raise TypeError("All entries in new_variable must be strings")
        logger.debug("new_variable is a list with %d items: %r", len(new_variable), new_variable)
    else:
        raise TypeError("new_variable must be a string or a list of strings")

    # chunk_grid can be a single grid or non-empty list of grids
    if isinstance(chunk_grid, list):
        if not chunk_grid:
            raise ValueError("chunk_grid list must not be empty")
        logger.debug("chunk_grid is a list with %d items, types: %s",
                     len(chunk_grid), [type(g) for g in chunk_grid])
    else:
        logger.debug("chunk_grid is a single item of type: %s", type(chunk_grid))

    # compressor can be None, a single compressor, or non-empty list
    if isinstance(compressor, list):
        if not compressor:
            raise ValueError("compressor list must not be empty")
        logger.debug("compressor is a list with %d items, types: %s",
                     len(compressor), [type(c) for c in compressor])
    elif compressor is None:
        logger.debug("compressor is None")
    else:
        logger.debug("compressor is a single item of type: %s", type(compressor))


def _normalize_new_variable(
    new_variable: str | list[str],
) -> list[str]:
    """Normalize new_variable to a list of names."""
    if isinstance(new_variable, str):
        result = [new_variable]
        logger.debug("Normalized new_variable from string %r to list: %r", new_variable, result)
        return result
    # At this point _validate_inputs already ensured this is non-empty list[str]
    result = list(new_variable)
    logger.debug("Normalized new_variable from list %r to list: %r", new_variable, result)
    return result


def _normalize_chunk_grid(
    chunk_grid: "RegularChunkGrid"
    | "RectilinearChunkGrid"
    | list["RegularChunkGrid" | "RectilinearChunkGrid"],
    num_variables: int,
) -> list["RegularChunkGrid" | "RectilinearChunkGrid"]:
    """Broadcast chunk_grid to match num_variables."""
    logger.debug("Normalizing chunk_grid for %d variables", num_variables)

    if isinstance(chunk_grid, list):
        logger.debug("Input chunk_grid is a list with %d items", len(chunk_grid))
        if len(chunk_grid) == 1 and num_variables > 1:
            result = chunk_grid * num_variables
            logger.debug("Broadcasting single chunk_grid to %d variables: %r", num_variables, result)
            return result
        if len(chunk_grid) == num_variables:
            result = list(chunk_grid)
            logger.debug("Using chunk_grid list as-is for %d variables: %r", num_variables, result)
            return result
        raise ValueError(
            "chunk_grid list length must be 1 or equal to the number of new variables"
        )
    # single grid reused for all variables
    result = [chunk_grid] * num_variables
    logger.debug("Replicating single chunk_grid for %d variables: %r", num_variables, result)
    return result


def _normalize_compressor(
    compressor: "ZFP" | "Blosc" | list["ZFP" | "Blosc"] | None,
    num_variables: int,
) -> list["ZFP" | "Blosc" | None]:
    """Broadcast compressor to match num_variables."""
    logger.debug("Normalizing compressor for %d variables", num_variables)

    if compressor is None:
        result = [None] * num_variables
        logger.debug("Setting compressor to None for all %d variables: %r", num_variables, result)
        return result

    if isinstance(compressor, list):
        logger.debug("Input compressor is a list with %d items", len(compressor))
        if len(compressor) == 1 and num_variables > 1:
            result = compressor * num_variables
            logger.debug("Broadcasting single compressor to %d variables: %r", num_variables, result)
            return result
        if len(compressor) == num_variables:
            result = list(compressor)
            logger.debug("Using compressor list as-is for %d variables: %r", num_variables, result)
            return result
        raise ValueError(
            "compressor list length must be 1 or equal to the number of new variables"
        )

    # single compressor reused for all variables
    result = [compressor] * num_variables
    logger.debug("Replicating single compressor for %d variables: %r", num_variables, result)
    return result


def from_variable(
    dataset_path: "UPath | Path | str",
    source_variable: str,
    new_variable: str | list[str],
    chunk_grid: "RegularChunkGrid"
    | "RectilinearChunkGrid"
    | list["RegularChunkGrid" | "RectilinearChunkGrid"],
    compressor: "ZFP" | "Blosc" | list["ZFP" | "Blosc"] | None = None,
    copy_metadata: bool = True,
) -> None:
    """Add new Variable(s) to the Dataset with different chunking and compression.

    Copies data from the source Variable to the new Variable(s) to create different
    access patterns.

    Args:
        dataset_path: The path to a pre-existing MDIO Dataset.
        source_variable: The name of the existing Variable to copy data from.
        new_variable: The name(s) of the new Variable(s) to create.
        chunk_grid:
            Chunk grid(s) to use for the new Variable(s).
            - Single grid: applied to all new variables.
            - List of grids: length must be 1 (broadcast) or match len(new_variable).
        compressor:
            Compressor(s) for the new Variable(s).
            - None: use source encoding compressor if present.
            - Single compressor: applied to all new variables.
            - List of compressors: length must be 1 (broadcast) or match len(new_variable).
        copy_metadata: Whether to copy attrs/encoding from the source Variable.
    """
    # 1) Basic validation (types, emptiness)
    logger.debug("Starting from_variable operation: dataset_path=%r, source_variable=%r",
                 dataset_path, source_variable)
    _validate_inputs(new_variable, chunk_grid, compressor)

    # 2) Normalize/broadcast each argument
    logger.debug("Normalizing inputs for processing")
    new_variables = _normalize_new_variable(new_variable)
    num_vars = len(new_variables)
    chunk_grids = _normalize_chunk_grid(chunk_grid, num_vars)
    compressors = _normalize_compressor(compressor, num_vars)
    logger.debug("After normalization: %d variables to create", num_vars)

    normed_path = _normalize_path(dataset_path)
    logger.debug("Opening dataset at: %r", normed_path)
    ds = open_mdio(normed_path)

    source_var = ds[source_variable]
    dims = source_var.dims
    shape = source_var.shape
    store_chunks = source_var.encoding.get("chunks", None)

    logger.debug("Source variable %r: dims=%r, shape=%r, store_chunks=%r",
                 source_variable, dims, shape, store_chunks)

    settings = MDIOSettings()
    num_workers = settings.export_cpus

    dask_config: dict[str, Any] = {"scheduler": "processes", "num_workers": num_workers}

    logger.debug("Using Dask config: %s", dask_config)

    # 3) One Dask config context, write each new variable sequentially
    with dask.config.set(**dask_config):
        logger.debug("Starting variable processing loop with Dask config")
        for name, grid, comp in tqdm(
            zip(new_variables, chunk_grids, compressors, strict=True),
            total=len(new_variables),
            desc="Generating newly chunked Variables",
            unit="variable"
        ):
            logger.debug("Processing variable %r with grid type %s and compressor type %s",
                         name, type(grid), type(comp))

            new_chunks = tuple(grid.configuration.chunk_shape)
            logger.debug("New chunk shape for variable %r: %r", name, new_chunks)

            if len(dims) != len(new_chunks):
                logger.warning(
                    "Original variable %r has dimensions %r, but new chunk shape %r "
                    "was provided for new variable %r. Behavior is currently undefined.",
                    source_variable,
                    dims,
                    new_chunks,
                    name,
                )

            # Build Dask chunk mapping for target chunks
            dest_mapping = dict(zip(dims, new_chunks, strict=True))
            logger.debug("Target chunk mapping: %r", dest_mapping)

            # Rechunk directly to target chunks - skip intermediate work chunks to avoid task explosion
            if store_chunks is not None and tuple(store_chunks) == new_chunks:
                logger.debug("Variable %r already has target chunks %r, using as-is", name, new_chunks)
                rechunked = source_var
            else:
                logger.debug("Rechunking variable %r directly to target chunks: %r", name, dest_mapping)
                rechunked = source_var.chunk(dest_mapping)
                logger.debug("Final rechunked array chunks: %r", rechunked.chunks)

            logger.debug("Dask task graph for variable %r has %d tasks", name, len(rechunked.__dask_graph__()))

            # Build DataArray for the new variable
            attrs = source_var.attrs.copy() if copy_metadata else {}
            logger.debug("Creating DataArray for variable %r with copy_metadata=%s", name, copy_metadata)
            new_da = DataArray(
                data=rechunked.data,
                dims=dims,
                coords=source_var.coords,
                attrs=attrs,
                name=name,
            )
            logger.debug("DataArray created with shape=%r, chunks=%r", new_da.shape, new_da.chunks)
            new_ds = new_da.to_dataset(name=name)
            logger.debug("Converted to dataset with %d variables", len(new_ds.data_vars))

            # Per-variable encoding
            encoding: dict[str, Any] = (
                source_var.encoding.copy() if copy_metadata else {}
            )
            encoding["chunks"] = new_chunks
            if comp is not None:
                compressor_encoding = _compressor_to_encoding(comp)
                encoding.update(compressor_encoding)
                logger.debug("Applied compressor encoding for variable %r: %r", name, compressor_encoding)
            new_ds[name].encoding = encoding
            logger.debug("Final encoding for variable %r: %r", name, encoding)

            # Clean up attrs that can conflict with consolidated metadata
            logger.debug("Cleaning up _FillValue attributes for variable %r", name)
            _remove_fillvalue_attrs(new_ds)

            # Drop non-dimensional coordinates to avoid chunk conflicts
            coords_to_drop = [coord for coord in new_ds.coords if coord not in new_ds.dims]
            if coords_to_drop:
                logger.debug("Dropping non-dimensional coordinates for variable %r: %r", name, coords_to_drop)
                new_ds = new_ds.drop_vars(coords_to_drop)
            else:
                logger.debug("No non-dimensional coordinates to drop for variable %r", name)

            logger.debug("Starting write operation for variable %r with compute=True", name)
            with TqdmCallback(desc=f"Writing variable '{name}'", unit="chunk"):
                to_mdio(new_ds, normed_path, mode="a", compute=True)
            logger.debug("Completed write operation for variable %r", name)

    logger.info(
        "Variable copy complete: %r -> %s",
        source_variable,
        ", ".join(new_variables),
    )
