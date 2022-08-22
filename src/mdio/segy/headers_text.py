"""Text header manipulation utilities."""


from __future__ import annotations

from typing import Sequence

import numpy as np

from mdio.segy.ebcdic import ASCII_TO_EBCDIC
from mdio.segy.ebcdic import EBCDIC_TO_ASCII


def wrap_strings(text_header: Sequence[str]) -> str:
    """Wrap list of strings into one.

    Args:
        text_header: list of str or tuple of str
            Text header in ASCII. List should contain N-strings where N is the
            number of lines in the text header. By default, SEG-Y text headers
            have 40 lines and 80 characters each.

    Returns:
        Concatenated string.
    """
    joined = "".join(text_header)
    return joined


def unwrap_string(text_header: str, rows: int = 40, cols: int = 80) -> list[str]:
    """Unwrap string into a list of strings with given rows and columns.

    Args:
        text_header: Text header in ASCII. Should be single string of text
            header. By default, SEG-Y text headers have 40 lines and 80
            characters each. A total of 3200 characters are expected,
            however this can be changed via the parameters `rows` and
            `columns` if needed.
        rows: Numbers of output rows. Default is 40. `rows` x `cols` must
            equal `len(text_header)`.
        cols: Number of output columns. Default is 80. `rows` x `cols` must
            equal `len(text_header)`.

    Returns:
        List of strings unwrapped to given specification.

    Raises:
        ValueError: if rows and columns don't match the size of string.
    """
    if rows * cols != len(text_header):
        raise ValueError("rows x cols must be equal text_header length.")

    unwrapped = []
    for idx in range(rows):
        start = idx * cols
        stop = start + cols
        unwrapped.append(text_header[start:stop])

    return unwrapped


def ascii_to_ebcdic(text_header: Sequence[str]) -> bytearray:
    """Convert ASCII encoded strings to EBCDIC bytearray.

    Args:
        text_header: Text header in ASCII. List should contain N-strings
            where N is the number of lines in the text header. By default,
            SEG-Y text headers have 40 lines and 80 characters each.
            Which totals 3200-bytes.

    Returns:
        EBCDIC encoded string as bytearray.
    """
    ascii_flat = wrap_strings(text_header)
    ascii_encoded = ascii_flat.encode()
    ascii_uint = np.frombuffer(ascii_encoded, dtype="uint8")
    ebcdic_uint = ASCII_TO_EBCDIC[ascii_uint]
    ebcdic_bytes = ebcdic_uint.tobytes()
    return ebcdic_bytes


def ebcdic_to_ascii(
    byte_string: bytearray | bytes,
    unwrap: bool = True,
    rows: int = 40,
    cols: int = 80,
) -> list[str] | str:
    """Convert EBCDIC encoded bytearray to ASCII string(s).

    Args:
        byte_string: EBCDIC encoded bytearray.
        unwrap: Option to unwrap the lines. Default is True.
        rows: Numbers of output rows. Default is 40. `rows` x `cols` must
            equal `len(text_header)`
        cols: Number of output columns. Default is 80. `rows` x `cols` must
            equal `len(text_header)`

    Returns:
        ASCII decoded text header.
    """
    length = len(byte_string)

    uint_repr = np.frombuffer(byte_string, dtype="uint8")
    ascii_bytes = EBCDIC_TO_ASCII[uint_repr].view(f"S{length}")
    ascii_str = ascii_bytes.astype("str")

    # Get rid of dimensions if not unwrapped
    ascii_str = np.array_str(np.squeeze(ascii_str))

    if unwrap is True:
        ascii_str = unwrap_string(ascii_str, rows, cols)

    return ascii_str
