"""Module for custom struct abstraction utilities."""


import sys
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class Dtype(Enum):
    """Dtype string to Numpy format enum."""

    STRING = ("STRING", "S")
    UINT8 = ("UINT8", "u1")
    UINT16 = ("UINT16", "u2")
    UINT32 = ("UINT32", "u4")
    UINT64 = ("UINT64", "u8")
    INT8 = ("INT8", "i1")
    INT16 = ("INT16", "i2")
    INT32 = ("INT32", "i4")
    INT64 = ("INT64", "i8")
    FLOAT16 = ("FLOAT16", "f2")
    FLOAT32 = ("FLOAT32", "f4")
    FLOAT64 = ("FLOAT64", "f8")
    IBM32 = ("IBM32", "u4")

    @property
    def numpy_dtype(self):
        """Return a numpy dtype of the Enum."""
        return np.dtype(self.value[1])


class ByteOrder(str, Enum):
    """Endianness string to Numpy format enum."""

    LITTLE = "<"
    BIG = ">"


@dataclass
class OrderedType:
    """Ordered Data Type (Format) abstraction.

    Args:
        type: Type definition.
        endian: Endianness of the data type.

    Attributes:
        dtype: Structured `numpy` data type.

    Methods:
        byteswap: Swaps endianness in place.
    """

    type: Dtype
    endian: ByteOrder

    def __len__(self) -> int:
        """Size of struct in bytes."""
        return self.dtype.itemsize

    @property
    def dtype(self):
        """Return Numpy dtype of the struct."""
        return np.dtype(self.endian + self.type.numpy_dtype)

    def byteswap(self):
        """Swap endianness in place."""
        swapped_dtype = self.dtype.newbyteorder()
        swapped_order = swapped_dtype.byteorder
        self.endian = ByteOrder(swapped_order)


SYS_BYTEORDER = ByteOrder[sys.byteorder.upper()]


def get_byteorder(array: NDArray) -> str:
    """Get byte order of numpy array.

    Args:
        array: Array like with `.dtype` attribute.

    Returns:
        String representing byte order in {"<", ">"}
    """
    if array.dtype.isnative:
        return SYS_BYTEORDER

    byteorder = array.dtype.byteorder

    return byteorder
