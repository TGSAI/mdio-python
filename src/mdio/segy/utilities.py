"""More utilities for reading SEG-Ys."""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from dask.array.core import normalize_chunks

from mdio.core import Dimension
from mdio.segy.geometry import GridOverrider
from mdio.segy.parsers import parse_headers

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from segy.arrays import HeaderArray

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.segy.file import SegyFileArguments
    from mdio.segy.file import SegyFileInfo

logger = logging.getLogger(__name__)


def get_grid_plan(  # noqa:  C901, PLR0912, PLR0913, PLR0915
    segy_file_kwargs: SegyFileArguments,
    segy_file_info: SegyFileInfo,
    chunksize: tuple[int, ...] | None,
    template: AbstractDatasetTemplate,
    return_headers: bool = False,
    grid_overrides: dict[str, Any] | None = None,
) -> tuple[list[Dimension], tuple[int, ...]] | tuple[list[Dimension], tuple[int, ...], HeaderArray]:
    """Infer dimension ranges, and increments.

    Generates multiple dimensions with the following steps:
    1. Read index headers
    2. Get min, max, and increments
    3. Create `Dimension` with appropriate range, index, and description.
    4. Create `Dimension` for sample axis using binary header.

    Args:
        segy_file_kwargs: SEG-Y file arguments.
        segy_file_info: SegyFileInfo instance containing the num_traces and sample_labels.
        chunksize:  Chunk sizes to be used in grid plan.
        template: MDIO template where coordinate names and domain will be taken.
        return_headers: Option to return parsed headers with `Dimension` objects. Default is False.
        grid_overrides: Option to add grid overrides. See main documentation.

    Returns:
        All index dimensions and chunksize or dimensions and chunksize together with header values.

    Raises:
        ValueError: If computed fields are not found after grid overrides.
    """
    if grid_overrides is None:
        grid_overrides = {}

    # Keep only dimension and non-dimension coordinates excluding the vertical axis
    horizontal_dimensions = template.spatial_dimension_names
    horizontal_coordinates = horizontal_dimensions + template.coordinate_names
    # Exclude calculated dimensions - they don't exist in SEG-Y headers
    calculated_dims = set(template.calculated_dimension_names)

    # Remove any to be computed fields - preserve order by using list comprehension instead of set operations
    computed_fields = set(template.calculated_dimension_names)
    horizontal_coordinates = tuple(c for c in horizontal_coordinates if c not in computed_fields)

    # Ensure non_binned_dims are included in the headers to parse, even if not in template
    if grid_overrides and "non_binned_dims" in grid_overrides:
        for dim in grid_overrides["non_binned_dims"]:
            if dim not in horizontal_coordinates:
                horizontal_coordinates = horizontal_coordinates + (dim,)

    headers_subset = parse_headers(
        segy_file_kwargs=segy_file_kwargs,
        num_traces=segy_file_info.num_traces,
        subset=tuple(c for c in horizontal_coordinates if c not in calculated_dims),
    )

    # Handle grid overrides.
    override_handler = GridOverrider()
    headers_subset, horizontal_coordinates, chunksize = override_handler.run(
        headers_subset,
        horizontal_coordinates,
        chunksize=chunksize,
        grid_overrides=grid_overrides,
        template=template,
    )

    # After grid overrides, determine final spatial dimensions and their chunk sizes
    non_binned_dims = set()
    if "NonBinned" in grid_overrides and "non_binned_dims" in grid_overrides:
        non_binned_dims = set(grid_overrides["non_binned_dims"])

    # Create mapping from dimension name to original chunk size for easy lookup
    original_spatial_dims = list(template.spatial_dimension_names)
    original_chunks = list(template.full_chunk_shape[:-1])  # Exclude vertical (sample/time) dimension
    dim_to_chunk = dict(zip(original_spatial_dims, original_chunks, strict=True))

    # Final spatial dimensions: keep trace and original dims, exclude non-binned dims
    final_spatial_dims = []
    final_spatial_chunks = []
    for name in horizontal_coordinates:
        if name in non_binned_dims:
            continue  # Skip dimensions that became coordinates
        if name == "trace":
            # Special handling for trace dimension
            chunk_val = int(grid_overrides.get("chunksize", 1)) if "NonBinned" in grid_overrides else 1
            final_spatial_dims.append(name)
            final_spatial_chunks.append(chunk_val)
        elif name in dim_to_chunk:
            # Use original chunk size for known dimensions
            final_spatial_dims.append(name)
            final_spatial_chunks.append(dim_to_chunk[name])

    if len(computed_fields) > 0 and not computed_fields.issubset(headers_subset.dtype.names):
        err = (
            f"Required computed fields {sorted(computed_fields)} for template {template.name} "
            f"not found after grid overrides. Please ensure correct overrides are applied."
        )
        raise ValueError(err)

    # Create dimensions from final_spatial_dims plus any computed fields that were added by grid overrides
    all_dimension_names = list(final_spatial_dims)
    added_computed_fields = []
    for computed_field in computed_fields:
        if computed_field in headers_subset.dtype.names and computed_field not in all_dimension_names:
            # Insert in template order
            if computed_field in template.spatial_dimension_names:
                insert_idx = template.spatial_dimension_names.index(computed_field)
                # Find position in all_dimension_names that corresponds to this template position
                actual_idx = min(insert_idx, len(all_dimension_names))
                all_dimension_names.insert(actual_idx, computed_field)
                # Track where we inserted and what chunk size it should have
                template_chunk_idx = template.spatial_dimension_names.index(computed_field)
                chunk_val = template.full_chunk_shape[template_chunk_idx]
                added_computed_fields.append((actual_idx, chunk_val))
            else:
                all_dimension_names.append(computed_field)
                added_computed_fields.append((len(all_dimension_names) - 1, 1))

    # Build chunksize including chunks for computed fields
    if added_computed_fields:
        chunk_list = list(final_spatial_chunks)
        for insert_idx, chunk_val in sorted(added_computed_fields, reverse=True):
            chunk_list.insert(insert_idx, chunk_val)
        chunksize = tuple(chunk_list + [template.full_chunk_shape[-1]])
    else:
        chunksize = tuple(final_spatial_chunks + [template.full_chunk_shape[-1]])

    dimensions = []
    for dim_name in all_dimension_names:
        if dim_name not in headers_subset.dtype.names:
            continue
        dim_unique = np.unique(headers_subset[dim_name])
        dimensions.append(Dimension(coords=dim_unique, name=dim_name))

    sample_labels = segy_file_info.sample_labels / 1000  # normalize

    if all(sample_labels.astype("int64") == sample_labels):
        sample_labels = sample_labels.astype("int64")

    vertical_dim = Dimension(coords=sample_labels, name=template.trace_domain)
    dimensions.append(vertical_dim)

    if return_headers:
        return dimensions, chunksize, headers_subset

    return dimensions, chunksize


def find_trailing_ones_index(dim_blocks: tuple[int, ...]) -> int:
    """Finds the index where trailing '1's begin in a tuple of dimension block sizes.

    If all values are '1', returns 0.

    Args:
        dim_blocks: A list of integers representing the data chunk dimensions.

    Returns:
        The index indicating the breakpoint where the trailing sequence of "1s"
        begins, or `0` if all values in the list are `1`.

    Examples:
        >>> find_trailing_ones_index((7, 5, 1, 1))
        2

        >>> find_trailing_ones_index((1, 1, 1, 1))
        0
    """
    total_dims = len(dim_blocks)
    trailing_ones = itertools.takewhile(lambda x: x == 1, reversed(dim_blocks))
    trailing_ones_count = sum(1 for _ in trailing_ones)

    return total_dims - trailing_ones_count


# TODO (Dmitriy Repin): Investigate the following warning generated at test_3d_export
# https://github.com/TGSAI/mdio-python/issues/657
# "The specified chunks separate the stored chunks along dimension "inline" starting at index 256.
# This could degrade performance. Instead, consider rechunking after loading."
def segy_export_rechunker(
    chunks: dict[str, int],
    sizes: dict[str, int],
    dtype: DTypeLike,
    limit: str = "300M",
) -> dict[str, int]:
    """Determine chunk sizes for writing out SEG-Y given limit.

    This module finds the desired chunk sizes for given chunk size `limit` in a depth first order.

    On disk chunks for MDIO are mainly optimized for visualization and ML applications. When we
    want to do export back to SEG-Y, it makes sense to have larger virtual chunks for processing
    of traces. We also recursively merge multiple files to reduce memory footprint.

    We choose to adjust chunks to be approx. 300 MB. We also need to do this in the order of
    fastest changing axis to the slowest changing axis becase the traces are expected to be
    serialized in the natural data order.

    Args:
        chunks: The chunk sizes on disk, per dimension.
        sizes: Shape of the whole array, per dimension.
        dtype: Numpy `dtype` of the array.
        limit: Chunk size limit in, optional. Default is "300 MB"

    Returns:
        Adjusted chunk sizes for further processing
    """
    dim_names = list(sizes.keys())
    sample_dim_key = dim_names[-1]

    # set sample dim chunks (last one) to max
    prev_chunks = chunks.copy()
    prev_chunks[sample_dim_key] = sizes[sample_dim_key]

    new_chunks = {}
    for dim_name in reversed(list(prev_chunks)):
        tmp_chunks: dict[str, int | str] = prev_chunks.copy()
        tmp_chunks[dim_name] = "auto"

        new_chunks = normalize_chunks(
            chunks=tuple(tmp_chunks.values()),
            shape=tuple(sizes.values()),
            limit=limit,
            previous_chunks=tuple(prev_chunks.values()),
            dtype=dtype,
        )
        new_chunks = dict(zip(dim_names, new_chunks, strict=True))
        prev_chunks = new_chunks.copy()

    # Ensure the sample (last dim) is single chunk.
    new_chunks[sample_dim_key] = sizes[sample_dim_key]
    logger.debug("Auto export rechunking to: %s", new_chunks)
    return new_chunks
