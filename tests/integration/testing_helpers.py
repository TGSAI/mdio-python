"""This module provides testing helpers for integration testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from segy.schema import HeaderField
from segy.schema import SegySpec

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr


def customize_segy_specs(
    segy_spec: SegySpec,
    index_bytes: tuple[int, ...] | None = None,
    index_names: tuple[int, ...] | None = None,
    index_types: tuple[int, ...] | None = None,
) -> SegySpec:
    """Customize SEG-Y specifications with user-defined index fields."""
    if not index_bytes:
        # No customization
        return segy_spec

    index_names = index_names or [f"dim_{i}" for i in range(len(index_bytes))]
    index_types = index_types or ["int32"] * len(index_bytes)

    if not (len(index_names) == len(index_bytes) == len(index_types)):
        err = "All index fields must have the same length."
        raise ValueError(err)

    # Index the dataset using a spec that interprets the user provided index headers.
    index_fields = []
    for name, byte, format_ in zip(index_names, index_bytes, index_types, strict=True):
        index_fields.append(HeaderField(name=name, byte=byte, format=format_))

    return segy_spec.customize(trace_header_fields=index_fields)


def get_values(arr: xr.DataArray) -> np.ndarray:
    """Extract actual values from an Xarray DataArray."""
    return arr.values


def get_inline_header_values(dataset: xr.Dataset) -> np.ndarray:
    """Extract a specific header value from an Xarray DataArray."""
    return dataset["inline"].values


def validate_variable(  # noqa PLR0913
    dataset: xr.Dataset,
    name: str,
    shape: list[int],
    dims: list[str],
    data_type: np.dtype,
    expected_values: range | None,
    actual_value_generator: Callable,
) -> None:
    """Validate the properties of a variable in an Xarray dataset."""
    arr = dataset[name]
    assert shape == arr.shape
    assert set(dims) == set(arr.dims)
    assert data_type == arr.dtype
    if expected_values is not None and actual_value_generator is not None:
        actual_values = actual_value_generator(arr)
        assert np.array_equal(expected_values, actual_values)
