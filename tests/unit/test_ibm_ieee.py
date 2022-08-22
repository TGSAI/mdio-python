"""Tests for IBM and IEEE floating point conversions.

Some references for test values
https://en.wikipedia.org/wiki/IBM_hexadecimal_floating-point
https://www.crewes.org/Documents/ResearchReports/2017/CRR201725.pdf
"""

import numpy as np
import pytest

from mdio.segy.ibm_float import byteswap_uint32
from mdio.segy.ibm_float import ibm2ieee
from mdio.segy.ibm_float import ieee2ibm


@pytest.mark.parametrize(
    "ieee, ibm",
    [
        ([0, -0.0], [0x00000000, 0x00000000]),
        ([0.1, -0.1], [0x40199999, 0xC0199999]),
        ([0.5, -0.5], [0x40800000, 0xC0800000]),
        ([1, -1], [0x41100000, 0xC1100000]),
        ([3.141593, -3.141593], [0x413243F7, 0xC13243F7]),
        ([0.15625, -0.15625], [0x40280000, 0xC0280000]),
        ([118.625, -118.625], [0x4276A000, 0xC276A000]),
        ([8521603, -8521603], [0x46820783, 0xC6820783]),
        ([3.4028235e38, -3.4028235e38], [0x60FFFFFF, 0xE0FFFFFF]),
    ],
)
class TestIbmIeee:
    """Test conversions, back and forth."""

    def test_ieee_to_ibm(self, ieee, ibm):
        """IEEE to IBM conversion."""
        ieee_fp32 = np.atleast_2d(np.float32(ieee))
        actual_ibm = np.squeeze(ieee2ibm(ieee_fp32))
        expected_ibm = np.squeeze(np.atleast_2d(np.uint32(ibm)))
        np.testing.assert_array_equal(actual_ibm, expected_ibm)

    def test_ibm_to_ieee(self, ieee, ibm):
        """IBM to IEEE conversion."""
        expected_ieee = np.asarray(ieee, dtype="float32")
        actual_ibm = np.asarray(ibm, dtype="uint32")

        # Assert up to 6 decimals (default)
        actual_ieee = ibm2ieee(actual_ibm)
        np.testing.assert_array_almost_equal(actual_ieee, expected_ieee)


@pytest.mark.parametrize("shape", [(1,), (10,), (20, 20), (150, 150)])
def test_ieee_to_ibm_roundtrip(shape: tuple):
    """IEEE to IBM and then back to IEEE conversion."""
    expected_ieee = np.random.randn(*shape).astype("float32")
    expected_ieee = np.atleast_2d(expected_ieee)

    actual_ibm = ieee2ibm(expected_ieee.copy())
    actual_ieee = ibm2ieee(actual_ibm)

    # Assert up to 6 decimals (default)
    np.testing.assert_array_almost_equal(actual_ieee, expected_ieee)


@pytest.mark.parametrize(
    "original, swapped",
    [
        (0x00000000, 0x00000000),
        (0xF0000000, 0x000000F0),
        (0x000FF000, 0x00F00F00),
        (0x0F0F0F0F, 0x0F0F0F0F),
        (0x0F0FF0F0F, 0x0F0FFFF0),
        (0xABCD1234, 0x3412CDAB),
        (0xA1B2C3D4, 0xD4C3B2A1),
        (0xAB12CD34, 0x34CD12AB),
    ],
)
def test_byteswap(original, swapped):
    """Test endian swapping operations."""
    original = np.uint32(original)
    expected = np.uint32(swapped)

    actual = byteswap_uint32(original)
    original_roundtrip = byteswap_uint32(actual)

    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(original, original_roundtrip)
