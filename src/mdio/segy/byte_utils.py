"""Module for custom struct abstraction utilities."""

import sys

from numpy.typing import NDArray
from segy.schema import Endianness


SYS_BYTEORDER = Endianness[sys.byteorder.upper()]


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
