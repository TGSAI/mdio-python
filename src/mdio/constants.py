"""Constant values used across MDIO."""


import numpy as np


FLOAT16_MAX = np.finfo("float16").max
FLOAT16_MIN = np.finfo("float16").min

FLOAT32_MAX = np.finfo("float32").max
FLOAT32_MIN = np.finfo("float32").min

FLOAT64_MIN = np.finfo("float64").min
FLOAT64_MAX = np.finfo("float64").max

INT8_MIN = -0x80
INT8_MAX = 0x7F

INT16_MIN = -0x8000
INT16_MAX = 0x7FFF

INT32_MIN = -0x80000000
INT32_MAX = 0x7FFFFFFF

UINT8_MIN = 0x0
UINT8_MAX = 0xFF

UINT16_MIN = 0x0
UINT16_MAX = 0xFFFF

UINT32_MIN = 0x0
UINT32_MAX = 0xFFFFFFFF
