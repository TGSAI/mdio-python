"""Module for custom struct abstraction utilities."""


from dataclasses import dataclass
from enum import Enum

import numpy as np


class Dtype(str, Enum):
    """Dtype string to Numpy format enum."""

    STRING = "S"
    UINT16 = "u2"
    UINT32 = "u4"
    UINT64 = "u8"
    INT16 = "i2"
    INT32 = "i4"
    INT64 = "i8"
    FLOAT16 = "f2"
    FLOAT32 = "f4"
    FLOAT64 = "f8"
    IBM32 = "u4"


class Endian(str, Enum):
    """Endianness string to Numpy format enum."""

    LITTLE = "<"
    BIG = ">"
    NATIVE = "|"


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
    endian: Endian

    def __len__(self) -> int:
        """Size of struct in bytes."""
        return self.dtype.itemsize

    @property
    def dtype(self):
        """Return Numpy dtype of the struct."""
        return np.dtype(self.endian + self.type)

    def byteswap(self):
        """Swap endianness in place."""
        swapped_dtype = self.dtype.newbyteorder()
        swapped_order = swapped_dtype.byteorder
        self.endian = Endian(swapped_order)
