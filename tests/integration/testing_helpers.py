"""This module provides testing helpers for integration testing."""

from collections.abc import Callable

import numpy as np
import xarray as xr

from mdio.builder.schemas.v1.units import AllUnitModel


def get_values(arr: xr.DataArray) -> np.ndarray:
    """Extract actual values from an Xarray DataArray."""
    return arr.values


def get_inline_header_values(dataset: xr.Dataset) -> np.ndarray:
    """Extract a specific header value from an Xarray DataArray."""
    return dataset["inline"].values


def validate_xr_variable(  # noqa PLR0913
    dataset: xr.Dataset,
    name: str,
    dims: dict[int],
    units: AllUnitModel,
    data_type: np.dtype,
    has_stats: bool = False,
    expected_values: range | None = None,
    actual_value_generator: Callable[[xr.DataArray], np.ndarray] | None = None,
) -> None:
    """Validate the properties of a variable in an Xarray dataset."""
    v = dataset[name]
    assert v is not None
    assert v.sizes == dims
    if hasattr(data_type, "fields") and data_type.fields is not None:
        # The following assertion will fail because of differences in offsets
        # assert data_type == arr.dtype

        # Compare field names
        expected_names = list(data_type.names)
        actual_names = list(v.dtype.names)
        assert expected_names == actual_names

        # Compare field types
        expected_types = [data_type[name] for name in data_type.names]
        actual_types = [v.dtype[name] for name in v.dtype.names]
        assert expected_types == actual_types
    else:
        assert data_type == v.dtype

    stats = v.attrs.get("statsV1", None)
    if has_stats:
        assert stats is not None, "StatsV1 should not be empty for dataset variables with stats"
    else:
        assert stats is None, "StatsV1 should be empty for dataset variables without stats"

    if units is not None:
        units_v1 = v.attrs.get("unitsV1", None)
        assert units_v1 is not None, "UnitsV1 should not be empty for dataset variables with units"
        assert units_v1 == units.model_dump(mode="json")
    else:
        assert "unitsV1" not in v.attrs, "UnitsV1 should not exist for unit-unaware variables"

    if expected_values is not None and actual_value_generator is not None:
        actual_values = actual_value_generator(v)
        assert np.array_equal(expected_values, actual_values)
