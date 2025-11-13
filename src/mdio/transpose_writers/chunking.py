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

    print(f"Original chunks: {original_chunks}")
    print(f"New chunks: {new_chunks}")
    print(f"Normalized chunks: {_normalize_chunks(original_chunks, new_chunks)}")

    
