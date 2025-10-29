"""This module provides testing helpers for integration testing."""

from collections.abc import Callable

import numpy as np
from numpy.typing import DTypeLike
import xarray as xr
from segy.schema import HeaderField
from segy.schema import ScalarType
from segy.schema.segy import SegySpec
from segy.standards import get_segy_standard

from mdio.builder.schemas.v1.units import AllUnitModel
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.schemas.v1.units import SpeedUnitEnum
from mdio.builder.schemas.v1.units import SpeedUnitModel
from mdio.builder.schemas.v1.units import TimeUnitEnum
from mdio.builder.schemas.v1.units import TimeUnitModel

UNITS_NONE = None
UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)
UNITS_MILLISECOND = TimeUnitModel(time=TimeUnitEnum.MILLISECOND)
UNITS_METERS_PER_SECOND = SpeedUnitModel(speed=SpeedUnitEnum.METERS_PER_SECOND)
UNITS_FOOT = LengthUnitModel(length=LengthUnitEnum.FOOT)
UNITS_FEET_PER_SECOND = SpeedUnitModel(speed=SpeedUnitEnum.FEET_PER_SECOND)


def get_teapot_segy_spec() -> SegySpec:
    """Return the customized SEG-Y specification for the teapot dome dataset."""
    teapot_fields = [
        HeaderField(name="inline", byte=17, format=ScalarType.INT32),
        HeaderField(name="crossline", byte=13, format=ScalarType.INT32),
        HeaderField(name="cdp_x", byte=81, format=ScalarType.INT32),
        HeaderField(name="cdp_y", byte=85, format=ScalarType.INT32),
    ]
    return get_segy_standard(1.0).customize(trace_header_fields=teapot_fields)


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
