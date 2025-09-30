"""Constant values used across MDIO."""

from enum import IntEnum

import numpy as np

from mdio.builder.schemas.dtype import ScalarType


class ZarrFormat(IntEnum):
    """Zarr version enum."""

    V2 = 2
    V3 = 3


FLOAT16_MAX = np.finfo("float16").max
FLOAT16_MIN = np.finfo("float16").min

FLOAT32_MAX = np.finfo("float32").max
FLOAT32_MIN = np.finfo("float32").min

FLOAT64_MIN = np.finfo("float64").min
FLOAT64_MAX = np.finfo("float64").max

INT8_MIN = np.iinfo("int8").min
INT8_MAX = np.iinfo("int8").max

INT16_MIN = np.iinfo("int16").min
INT16_MAX = np.iinfo("int16").max

INT32_MIN = np.iinfo("int32").min
INT32_MAX = np.iinfo("int32").max

INT64_MIN = np.iinfo("int64").min
INT64_MAX = np.iinfo("int64").max

UINT8_MIN = 0
UINT8_MAX = np.iinfo("uint8").max

UINT16_MIN = 0
UINT16_MAX = np.iinfo("uint16").max

UINT32_MIN = 0
UINT32_MAX = np.iinfo("uint32").max

UINT64_MIN = 0
UINT64_MAX = np.iinfo("uint64").max

# Zarr fill values for different scalar types
fill_value_map = {
    ScalarType.BOOL: None,
    ScalarType.FLOAT16: np.nan,
    ScalarType.FLOAT32: np.nan,
    ScalarType.FLOAT64: np.nan,
    ScalarType.UINT8: UINT8_MAX,
    ScalarType.UINT16: UINT16_MAX,
    ScalarType.UINT32: UINT32_MAX,
    ScalarType.UINT64: UINT64_MAX,
    ScalarType.INT8: INT8_MAX,
    ScalarType.INT16: INT16_MAX,
    ScalarType.INT32: INT32_MAX,
    ScalarType.INT64: INT64_MAX,
    ScalarType.COMPLEX64: complex(np.nan, np.nan),
    ScalarType.COMPLEX128: complex(np.nan, np.nan),
    ScalarType.COMPLEX256: complex(np.nan, np.nan),
    ScalarType.BYTES240: b"\x00" * 240,
}
