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


def _normalize_chunks(
    original_chunks: tuple[int, ...] | None,
    target_chunks: tuple[int, ...],
) -> tuple[int, ...]:
    """Choose a 'work chunk' shape per dim from original and target.

    If original_chunks is known, pick max(original, target) per dimension.
    Otherwise, just use target_chunks.
    """
    if original_chunks is None:
        return target_chunks

    return tuple(
        max(o, t) for o, t in zip(original_chunks, target_chunks, strict=True)
    )


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

    # new_variable must be str or non-empty list[str]
    if isinstance(new_variable, str):
        pass
    elif isinstance(new_variable, list):
        if not new_variable:
            raise ValueError("new_variable list must not be empty")
        if not all(isinstance(v, str) for v in new_variable):
            raise TypeError("All entries in new_variable must be strings")
    else:
        raise TypeError("new_variable must be a string or a list of strings")

    # chunk_grid can be a single grid or non-empty list of grids
    if isinstance(chunk_grid, list) and not chunk_grid:
        raise ValueError("chunk_grid list must not be empty")

    # compressor can be None, a single compressor, or non-empty list
    if isinstance(compressor, list) and not compressor:
        raise ValueError("compressor list must not be empty")


def _normalize_new_variable(
    new_variable: str | list[str],
) -> list[str]:
    """Normalize new_variable to a list of names."""
    if isinstance(new_variable, str):
        return [new_variable]
    # At this point _validate_inputs already ensured this is non-empty list[str]
    return list(new_variable)


def _normalize_chunk_grid(
    chunk_grid: "RegularChunkGrid"
    | "RectilinearChunkGrid"
    | list["RegularChunkGrid" | "RectilinearChunkGrid"],
    num_variables: int,
) -> list["RegularChunkGrid" | "RectilinearChunkGrid"]:
    """Broadcast chunk_grid to match num_variables."""
    if isinstance(chunk_grid, list):
        if len(chunk_grid) == 1 and num_variables > 1:
            return chunk_grid * num_variables
        if len(chunk_grid) == num_variables:
            return list(chunk_grid)
        raise ValueError(
            "chunk_grid list length must be 1 or equal to the number of new variables"
        )
    # single grid reused for all variables
    return [chunk_grid] * num_variables


def _normalize_compressor(
    compressor: "ZFP" | "Blosc" | list["ZFP" | "Blosc"] | None,
    num_variables: int,
) -> list["ZFP" | "Blosc" | None]:
    """Broadcast compressor to match num_variables."""
    if compressor is None:
        return [None] * num_variables

    if isinstance(compressor, list):
        if len(compressor) == 1 and num_variables > 1:
            return compressor * num_variables
        if len(compressor) == num_variables:
            return list(compressor)
        raise ValueError(
            "compressor list length must be 1 or equal to the number of new variables"
        )

    # single compressor reused for all variables
    return [compressor] * num_variables


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
    _validate_inputs(new_variable, chunk_grid, compressor)

    # 2) Normalize/broadcast each argument
    new_variables = _normalize_new_variable(new_variable)
    num_vars = len(new_variables)
    chunk_grids = _normalize_chunk_grid(chunk_grid, num_vars)
    compressors = _normalize_compressor(compressor, num_vars)

    normed_path = _normalize_path(dataset_path)
    ds = open_mdio(normed_path)

    source_var = ds[source_variable]
    dims = source_var.dims

    store_chunks = source_var.encoding.get("chunks", None)

    settings = MDIOSettings()
    num_workers = settings.export_cpus

    dask_config: dict[str, Any] = {"scheduler": "processes", "num_workers": num_workers}

    logger.debug("Using Dask config: %s", dask_config)

    # 3) One Dask config context, write each new variable sequentially
    with dask.config.set(**dask_config):
        for name, grid, comp in tqdm(
            zip(new_variables, chunk_grids, compressors, strict=True),
            total=len(new_variables),
            desc="Generating newly chunked Variables",
            unit="variable"
        ):
            new_chunks = tuple(grid.configuration.chunk_shape)

            if len(dims) != len(new_chunks):
                logger.warning(
                    "Original variable %r has dimensions %r, but new chunk shape %r "
                    "was provided for new variable %r. Behavior is currently undefined.",
                    source_variable,
                    dims,
                    new_chunks,
                    name,
                )

            # Compute a 'work chunk' that is compatible with both source and dest
            work_chunks = _normalize_chunks(store_chunks, new_chunks)

            # Build Dask chunk mappings
            work_mapping = dict(zip(dims, work_chunks, strict=True))
            dest_mapping = dict(zip(dims, new_chunks, strict=True))

            # Step 1: chunk to 'work' shape if needed
            # If store_chunks is None, this will just chunk from whatever dask sees
            if store_chunks is not None and tuple(store_chunks) == work_chunks:
                source_work = source_var
            else:
                source_work = source_var.chunk(work_mapping)

            # Step 2: ensure final chunks match destination encoding
            if work_chunks == new_chunks:
                rechunked = source_work
            else:
                rechunked = source_work.chunk(dest_mapping)

            # Build DataArray for the new variable
            attrs = source_var.attrs.copy() if copy_metadata else {}
            new_da = DataArray(
                data=rechunked.data,
                dims=dims,
                coords=source_var.coords,
                attrs=attrs,
                name=name,
            )
            new_ds = new_da.to_dataset(name=name)

            # Per-variable encoding
            encoding: dict[str, Any] = (
                source_var.encoding.copy() if copy_metadata else {}
            )
            encoding["chunks"] = new_chunks
            if comp is not None:
                encoding.update(_compressor_to_encoding(comp))
            new_ds[name].encoding = encoding

            # Clean up attrs that can conflict with consolidated metadata
            _remove_fillvalue_attrs(new_ds)

            # Drop non-dimensional coordinates to avoid chunk conflicts
            coords_to_drop = [coord for coord in new_ds.coords if coord not in new_ds.dims]
            if coords_to_drop:
                new_ds = new_ds.drop_vars(coords_to_drop)

            with TqdmCallback(desc=f"Writing variable '{name}'", unit="chunk"):
                to_mdio(new_ds, normed_path, mode="a", compute=True)

    logger.info(
        "Variable copy complete: %r -> %s",
        source_variable,
        ", ".join(new_variables),
    )
