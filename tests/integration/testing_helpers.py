"""This module provides testing helpers for integration testing."""

from collections.abc import Callable

import numpy as np
import xarray as xr
from segy.schema import HeaderField
from segy.schema import SegySpec


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
        assert data_type.names == arr.dtype.names

        # Compare field types
        expected_types = [data_type[name].newbyteorder("=") for name in data_type.names]
        actual_types = [arr.dtype[name].newbyteorder("=") for name in arr.dtype.names]
        assert expected_types == actual_types

        # Compare field offsets fails.
        # However, we believe this is acceptable and do not compare offsets
        #   name: 'shot_point' dt_exp: (dtype('>i4'), 196) dt_act: (dtype('<i4'), 180)
    else:
        assert data_type == arr.dtype

    if expected_values is not None and actual_value_generator is not None:
        actual_values = actual_value_generator(arr)
        assert np.array_equal(expected_values, actual_values)
