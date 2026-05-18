"""Shared builders for ingestion unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mdio.core.dimension import Dimension
from mdio.core.grid import Grid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def make_grid(dim_specs: list[tuple[str, NDArray]]) -> Grid:
    """Build a Grid from a list of ``(name, coords)`` pairs."""
    dims = [Dimension(coords=coords, name=name) for name, coords in dim_specs]
    return Grid(dims=dims)


def make_grid_with_map(
    dim_specs: list[tuple[str, NDArray]],
    live_records: Sequence[tuple],
) -> Grid:
    """Build a Grid and populate its trace map via ``Grid.build_map``.

    The trace index for each record matches its position in ``live_records``, exactly
    mirroring how production ingestion code assigns trace ordinals when streaming
    SEG-Y headers through ``Grid.build_map``.

    Args:
        dim_specs: Ordered ``(name, coords)`` pairs. The last entry is the vertical
            (sample/depth) dimension and is excluded from the trace map per
            ``Grid.build_map`` conventions.
        live_records: Per-trace tuples giving the value of each non-sample
            dimension, in dimension order. Cells absent from this list remain at
            the map's fill value.

    Returns:
        Grid with a real Zarr-backed ``map`` and ``live_mask`` populated.
    """
    grid = make_grid(dim_specs)
    non_sample_dims = grid.dims[:-1]
    names = [d.name for d in non_sample_dims]
    formats = [np.asarray(d.coords).dtype for d in non_sample_dims]
    header_dtype = np.dtype({"names": names, "formats": formats})
    headers = np.empty(len(live_records), dtype=header_dtype)
    for idx, values in enumerate(live_records):
        for name, value in zip(names, values, strict=True):
            headers[name][idx] = value
    grid.build_map(headers)
    return grid


def make_header_array(field_values: dict[str, NDArray]) -> NDArray:
    """Build a structured numpy array mimicking a SEG-Y HeaderArray.

    Args:
        field_values: Mapping of field name to a 1-D array of values. All arrays must
            share the same shape.

    Returns:
        Structured ``ndarray`` with one named column per field.
    """
    sample = next(iter(field_values.values()))
    dtype = np.dtype({"names": list(field_values), "formats": [v.dtype for v in field_values.values()]})
    arr = np.empty(sample.shape, dtype=dtype)
    for name, values in field_values.items():
        arr[name] = values
    return arr
