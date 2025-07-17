"""Constant values used across MDIO."""

from numpy import finfo as np_finfo
from numpy import iinfo as np_iinfo
from numpy import nan as np_nan

from mdio.schemas.dtype import ScalarType

FLOAT16_MAX = np_finfo("float16").max
FLOAT16_MIN = np_finfo("float16").min

FLOAT32_MAX = np_finfo("float32").max
FLOAT32_MIN = np_finfo("float32").min

FLOAT64_MIN = np_finfo("float64").min
FLOAT64_MAX = np_finfo("float64").max

INT8_MIN = np_iinfo("int8").min
INT8_MAX = np_iinfo("int8").max

INT16_MIN = np_iinfo("int16").min
INT16_MAX = np_iinfo("int16").max

INT32_MIN = np_iinfo("int32").min
INT32_MAX = np_iinfo("int32").max

INT64_MIN = np_iinfo("int64").min
INT64_MAX = np_iinfo("int64").max

UINT8_MIN = 0
UINT8_MAX = np_iinfo("uint8").max

UINT16_MIN = 0
UINT16_MAX = np_iinfo("uint16").max

UINT32_MIN = 0
UINT32_MAX = np_iinfo("uint32").max

UINT64_MIN = 0
UINT64_MAX = np_iinfo("uint64").max

# Zarr fill values for different scalar types
fill_value_map = {
    ScalarType.BOOL: None,
    ScalarType.FLOAT16: np_nan,
    ScalarType.FLOAT32: np_nan,
    ScalarType.FLOAT64: np_nan,
    ScalarType.UINT8: UINT8_MAX,
    ScalarType.UINT16: UINT16_MAX,
    ScalarType.UINT32: UINT32_MAX,
    ScalarType.UINT64: UINT64_MAX,
    ScalarType.INT8: INT8_MAX,
    ScalarType.INT16: INT16_MAX,
    ScalarType.INT32: INT32_MAX,
    ScalarType.INT64: INT64_MAX,
    ScalarType.COMPLEX64: complex(np_nan, np_nan),
    ScalarType.COMPLEX128: complex(np_nan, np_nan),
    ScalarType.COMPLEX256: complex(np_nan, np_nan),
}
