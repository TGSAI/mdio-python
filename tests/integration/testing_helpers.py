"""This module provides testing helpers for integration testing."""

from collections.abc import Callable

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from segy.schema import HeaderField
from segy.schema import SegySpec


def customize_segy_specs(
    segy_spec: SegySpec,
    index_bytes: tuple[int, ...] | None = None,
    index_names: tuple[int, ...] | None = None,
    index_types: tuple[int, ...] | None = None,
    keep_unaltered: bool = False,
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

    fields = {}
    if keep_unaltered:
        # Keep unaltered fields, but remove the fields that are being customized
        # to avoid field name duplication
        for f in segy_spec.trace.header.fields:
            if f.name not in index_names:
                fields[f.byte] = f

    # Index the dataset using a spec that interprets the user provided index headers.
    for name, byte, format_ in zip(index_names, index_bytes, index_types, strict=True):
        fields[byte] = HeaderField(name=name, byte=byte, format=format_)

    return segy_spec.customize(trace_header_fields=fields.values())


def get_values(arr: xr.DataArray) -> np.ndarray:
    """Extract actual values from an Xarray DataArray."""
    return arr.values


def get_inline_header_values(dataset: xr.Dataset) -> np.ndarray:
    """Extract a specific header value from an Xarray DataArray."""
    return dataset["inline"].values


def validate_variable(  # noqa PLR0913
    dataset: xr.Dataset,
    name: str,
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    data_type: DTypeLike,
    expected_values: range | None,
    actual_value_generator: Callable[[xr.DataArray], np.ndarray] | None = None,
) -> None:
    """Validate the properties of a variable in an Xarray dataset."""
    arr = dataset[name]
    assert shape == arr.shape
    assert set(dims) == set(arr.dims)
    if hasattr(data_type, "fields") and data_type.fields is not None:
        # The following assertion will fail because of differences in offsets
        # assert data_type == arr.dtype

        # Compare field names
        expected_names = list(data_type.names)
        actual_names = list(arr.dtype.names)
        assert expected_names == actual_names

        # Compare field types
        expected_types = [data_type[name] for name in data_type.names]
        actual_types = [arr.dtype[name] for name in arr.dtype.names]
        assert expected_types == actual_types
    else:
        assert data_type == arr.dtype

    if expected_values is not None and actual_value_generator is not None:
        actual_values = actual_value_generator(arr)
        assert np.array_equal(expected_values, actual_values)
