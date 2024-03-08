"""Tests for lower level text header modules."""

import random
import string
from binascii import hexlify
from binascii import unhexlify

import pytest

from mdio.seismic.headers_text import ascii_to_ebcdic
from mdio.seismic.headers_text import ebcdic_to_ascii
from mdio.seismic.headers_text import unwrap_string
from mdio.seismic.headers_text import wrap_strings


@pytest.mark.parametrize(
    "ascii_str, hex_stream",
    [
        ("f", b"86"),
        ("~", b"a1"),
        ("aA", b"81-c1"),
        ("zZ", b"a9-e9"),
        ("@Ac5", b"7c-c1-83-f5"),
        (["# 45", "13.2", "J|kX"], b"7b-40-f4-f5-f1-f3-4b-f2-d1-6a-92-e7"),
    ],
)
class TestEbcdic:
    """Test ASCII/EBCDIC coding."""

    def test_encode(self, ascii_str: str, hex_stream: bytes) -> None:
        """Encoding from ASCII to EBCDIC."""
        ebcdic_buffer = ascii_to_ebcdic(ascii_str)
        ascii_to_hex = hexlify(ebcdic_buffer, sep="-")
        assert ascii_to_hex == hex_stream

    def test_decode(self, ascii_str: str, hex_stream: bytes) -> None:
        """Decoding from EBCDIC to ASCII."""
        ebcdic_buffer = unhexlify(hex_stream.replace(b"-", b""))

        if isinstance(ascii_str, list):
            rows = len(ascii_str)
            cols = len(ascii_str[0])
        else:
            rows, cols = 1, len(ascii_str)

        ascii_str = ebcdic_to_ascii(ebcdic_buffer, unwrap=True, rows=rows, cols=cols)
        print(ascii_str)


@pytest.mark.parametrize(
    ("rows", "cols", "wrapped", "unwrapped"),
    [
        (1, 1, "a", ["a"]),
        (1, 2, "12", ["12"]),
        (2, 3, "ab~|ef", ["ab~", "|ef"]),
        (4, 3, "a1b2c3d4e5f6", ["a1b", "2c3", "d4e", "5f6"]),
    ],
)
class TestWrapping:
    """String wrappers and unwrappers."""

    def test_wrap(
        self, rows: int, cols: int, wrapped: str, unwrapped: list[str]
    ) -> None:
        """Test from list of strings to a flat string."""
        actual = wrap_strings(unwrapped)
        assert wrapped == actual
        assert len(wrapped) == rows * cols

    def test_unwrap(
        self, rows: int, cols: int, wrapped: str, unwrapped: list[str]
    ) -> None:
        """Test flat string to list of strings."""
        actual = unwrap_string(wrapped, rows, cols)
        assert unwrapped == actual
        assert len(unwrapped) == rows
        assert len(unwrapped[0]) == cols

    def test_wrong_length_exception(
        self, rows: int, cols: int, wrapped: str, unwrapped: list[str]
    ) -> None:
        """Test if we raise the exception when rows/cols don't match size."""
        rows += 1
        cols += 1

        msg = "rows x cols must be equal text_header length"
        with pytest.raises(ValueError, match=msg):
            assert unwrapped == unwrap_string(wrapped, rows, cols)


@pytest.mark.parametrize(
    (
        "rows",
        "cols",
        "uppercase_prob",
        "lowercase_prob",
        "digits_prob",
        "symbol_prob",
        "space_prob",
    ),
    [
        (40, 80, 0.5, 1, 0.25, 0.05, 10),
        (15, 25, 0.2, 1.5, 0.55, 0.1, 5),
        (30, 60, 0.3, 1.2, 0.35, 0.15, 6),
    ],
)
def test_roundtrip(  # noqa: PLR0913
    rows: int,
    cols: int,
    uppercase_prob: float,
    lowercase_prob: float,
    digits_prob: float,
    symbol_prob: float,
    space_prob: float,
) -> None:
    """Test converting randomly generated ASCII / EBCDIC and back."""
    # Get all kinds of letters, numbers, and symbols
    upper = string.ascii_uppercase
    lower = string.ascii_lowercase
    digits = string.digits
    symbols = string.punctuation
    whitespace = " "

    # Define probabilities (weights) for each character for random sampling
    upper_prob = [uppercase_prob] * len(upper)
    lower_prob = [lowercase_prob] * len(lower)
    digits_prob = [digits_prob] * len(digits)
    symbol_prob = [symbol_prob] * len(symbols)
    space_prob = [space_prob] * len(whitespace)

    all_chars = upper + lower + digits + symbols + whitespace
    all_probs = upper_prob + lower_prob + digits_prob + symbol_prob + space_prob

    # Generate an "unwrapped" text header in ASCII
    text_header = []
    for idx in range(1, rows + 1):
        line_header = r"C{line_no:02d}"
        random_string = random.choices(  # noqa: S311
            all_chars, weights=all_probs, k=cols - 4
        )
        random_string = r"".join(random_string)
        line_data = r" ".join([line_header.format(line_no=idx), random_string])
        text_header.append(line_data)

    # Make EBCDIC, then come back to ASCII
    ebcdic_buf = ascii_to_ebcdic(text_header)
    round_ascii = ebcdic_to_ascii(ebcdic_buf, unwrap=False)
    round_ascii_unwrap = ebcdic_to_ascii(ebcdic_buf, unwrap=True, rows=rows, cols=cols)

    # Testing both wrapped and unwrapped
    assert wrap_strings(text_header) == round_ascii
    assert text_header == round_ascii_unwrap
