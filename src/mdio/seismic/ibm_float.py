"""Low-level floating point conversion operations."""


import os

import numba as nb
import numpy as np


# If Numba's JIT compilation is disabled, force vectorized
# functions to Numba's Object Mode. This is only used when running
# tests and we want a coverage report to JIT functions.
# In every other case, functions will be JIT compiled.
try:
    NUMBA_DISABLE_JIT = os.environ["NUMBA_DISABLE_JIT"]
except KeyError:  # pragma: no cover
    NUMBA_DISABLE_JIT = 0

OBJECT_MODE = True if NUMBA_DISABLE_JIT else False
JIT_CACHE = False if NUMBA_DISABLE_JIT else True
JIT_TARGET = "cpu"
JIT_KWARGS = dict(cache=JIT_CACHE, forceobj=OBJECT_MODE)

# IEEE to IBM MASKS ETC
IEEE32_SIGN = np.uint32(0x80000000)
IEEE32_EXPONENT = np.int32(0x7F800000)
IEEE32_FRACTION = np.uint32(0x7FFFFF)

# IBM to IEEE MASKS ETC
BASE2POW24 = np.uint32(0x1000000)
IBM32_EXPONENT = np.uint32(0x7F000000)
IBM32_FRACTION = np.uint32(0xFFFFFF)

# For Byte Swapping
BYTEMASK_1_3 = np.uint32(0xFF00FF00)
BYTEMASK_2_4 = np.uint32(0xFF00FF)


@nb.njit(
    "uint32(float32)",
    cache=JIT_CACHE,
    locals={
        "sign": nb.uint32,
        "exponent": nb.int32,
        "exp_remainder": nb.int8,
        "downshift": nb.int8,
        "ibm_mantissa": nb.int32,
    },
)
def ieee2ibm_single(ieee: np.float32) -> np.uint32:
    """IEEE Float to IBM Float conversion.

    Modified from here:
    https://mail.python.org/pipermail/scipy-user/2011-June/029661.html

    Had to do some CPU and memory optimizations + Numba JIT compilation

    Assuming `ieee_array` is little endian and float32. Will convert to float32 if not.
    Returns `ibm_array` as little endian too.

    Byte swapping is up to user after this function.

    Args:
        ieee: Numpy IEEE 32-bit float array.

    Returns:
        IBM 32-bit float converted array with int32 view.
    """
    ieee = np.float32(ieee).view(np.uint32)

    if ieee in [0, 2147483648]:  # 0.0 or np.float32(-0.0).view('uint32')
        return 0

    # Get IEEE's sign and exponent
    sign = ieee & IEEE32_SIGN
    exponent = ((ieee & IEEE32_EXPONENT) >> 23) - 127
    # The IBM 7-bit exponent is to the base 16 and the mantissa is presumed to
    # be entirely to the right of the radix point. In contrast, the IEEE
    # exponent is to the base 2 and there is an assumed 1-bit to the left of
    # the radix point.
    # Note: reusing exponent variable, -> it is actually exp16

    # exp16, exp_remainder
    exponent, exp_remainder = divmod(exponent + 1, 4)
    exponent += exp_remainder != 0
    downshift = 4 - exp_remainder if exp_remainder else 0
    exponent = exponent + 64
    # From here down exponent -> ibm_exponent
    exponent = 0 if exponent < 0 else exponent
    exponent = 127 if exponent > 127 else exponent
    exponent = exponent << 24
    exponent = exponent if ieee else 0

    # Add the implicit initial 1-bit to the 23-bit IEEE mantissa to get the
    # 24-bit IBM mantissa. Downshift it by the remainder from the exponent's
    # division by 4. It is allowed to have up to 3 leading 0s.
    ibm_mantissa = ((ieee & IEEE32_FRACTION) | 0x800000) >> downshift
    ibm = sign | exponent | ibm_mantissa

    return ibm


@nb.njit(
    "float32(uint32)",
    cache=JIT_CACHE,
    locals={
        "sign_bit": nb.boolean,
        "sign": nb.int8,
        "exponent": nb.uint8,
        "mantissa": nb.float32,
        "ieee": nb.float32,
    },
)
def ibm2ieee_single(ibm: np.uint32) -> np.float32:
    """Converts a 32-bit IBM floating point number into 32-bit IEEE format.

    https://en.wikipedia.org/wiki/IBM_hexadecimal_floating-point
    https://en.wikipedia.org/wiki/IEEE_754

    FP Number = (IBM) -1**sign_bit × 0.significand × 16**(exponent−64)

    Args:
        ibm: Value in 32-bit IBM Float in Little-Endian Format.

    Returns:
        Value parsed to 32-bit IEEE Float in Little-Endian Format.
    """
    if ibm & IBM32_FRACTION == 0:
        return 0.0

    sign_bit = ibm >> 31

    exponent = (ibm & IBM32_EXPONENT) >> 24
    mantissa = (ibm & IBM32_FRACTION) / BASE2POW24

    # (1 - 2 * sign_bit) is about 50x faster than (-1)**sign_bit
    sign = 1 - 2 * sign_bit

    # This 16.0 (instead of just 16) is super important.
    # If the base is not a float, it won't work for negative
    # exponents, and fail silently and return zero.
    ieee = sign * mantissa * 16.0 ** (exponent - 64)

    return ieee


@nb.njit("uint32(uint32)", cache=JIT_CACHE)
def byteswap_uint32_single(value):
    """Endianness swapping that can be JIT compiled.

    This is faster or on par with the numpy implementation depending
    on the size of the array.

    We first shift (4, 3, 2, 1) to (3, 4, 1, 2)
    Then shift (3, 4, 3, 2) to (1, 2, 3, 4)

    Which yields (4, 3, 2, 1) -> (1, 2, 3, 4) or vice-versa.

    Args:
        value: Value to be byte-swapped.

    Returns:
        Byte-swapped value in same dtype.
    """
    value = np.uint32(value)

    if value == 0:
        return value

    value = ((value << 8) & BYTEMASK_1_3) | ((value >> 8) & BYTEMASK_2_4)
    value = np.uint32(value << 16) | np.uint32(value >> 16)
    return value


@nb.vectorize("uint32(float32)", target=JIT_TARGET, **JIT_KWARGS)
def ieee2ibm(ieee_array: np.float32) -> np.uint32:  # pragma: no cover
    """Wrapper for vectorizing IEEE to IBM conversion to arrays."""
    return ieee2ibm_single(ieee_array)


@nb.vectorize("float32(uint32)", target=JIT_TARGET, **JIT_KWARGS)
def ibm2ieee(ibm_array: np.uint32) -> np.float32:  # pragma: no cover
    """Wrapper for vectorizing IBM to IEEE conversion to arrays."""
    return ibm2ieee_single(ibm_array)


@nb.vectorize("uint32(uint32)", **JIT_KWARGS)
def byteswap_uint32(value):  # pragma: no cover
    """Wrapper for vectorizing byte-swap to arrays."""
    return byteswap_uint32_single(value)
