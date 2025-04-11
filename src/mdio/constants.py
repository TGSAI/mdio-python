"""Constant values used across MDIO."""

import numpy as np


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
