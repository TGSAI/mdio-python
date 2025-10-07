"""Tests for coordinate scalar getters and apply functions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
from segy import SegyFile
from segy.standards import SegyStandard
from segy.standards.fields import trace as trace_header_fields

from mdio.segy.scalar import _apply_coordinate_scalar
from mdio.segy.scalar import _get_coordinate_scalar

if TYPE_CHECKING:
    from numpy.typing import NDArray

COORD_SCALAR_KEY = trace_header_fields.Rev0.COORDINATE_SCALAR.model.name


@pytest.fixture
def mock_segy_file() -> SegyFile:
    """Mock SegyFile object."""
    segy_file = MagicMock(spec=SegyFile)
    segy_file.spec = MagicMock()
    segy_file.header = [MagicMock()]
    return segy_file


@pytest.mark.parametrize("scalar", [1, 100, 10000, -10, -1000])
def test_get_coordinate_scalar_valid(mock_segy_file: SegyFile, scalar: int) -> None:
    """Test valid options when getting coordinate scalar."""
    mock_segy_file.spec.segy_standard = SegyStandard.REV1
    mock_segy_file.header[0].__getitem__.return_value = scalar

    result = _get_coordinate_scalar(mock_segy_file)

    assert result == scalar


@pytest.mark.parametrize(
    "revision",
    [SegyStandard.REV2, SegyStandard.REV21],
)
def test_get_coordinate_scalar_zero_rev2_plus(mock_segy_file: SegyFile, revision: SegyStandard) -> None:
    """Test when scalar is normalized to 1 (from 0) in Rev2+."""
    mock_segy_file.spec.segy_standard = revision
    mock_segy_file.header[0].__getitem__.return_value = 0

    result = _get_coordinate_scalar(mock_segy_file)

    assert result == 1


@pytest.mark.parametrize(
    ("scalar", "revision", "error_msg"),
    [
        (0, SegyStandard.REV0, "Invalid coordinate scalar: 0 for file revision SegyStandard.REV0."),
        (110, SegyStandard.REV1, "Invalid coordinate scalar: 110 for file revision SegyStandard.REV1."),
        (32768, SegyStandard.REV1, "Invalid coordinate scalar: 32768 for file revision SegyStandard.REV1."),
    ],
)
def test_get_coordinate_scalar_invalid(
    mock_segy_file: SegyFile, scalar: int, revision: SegyStandard, error_msg: str
) -> None:
    """Test invalid options when getting coordinate scalar."""
    mock_segy_file.spec.segy_standard = revision
    mock_segy_file.header[0].__getitem__.return_value = scalar

    with pytest.raises(ValueError, match=error_msg):
        _get_coordinate_scalar(mock_segy_file)


@pytest.mark.parametrize(
    ("data", "scalar", "expected"),
    [
        # POSITIVE
        (np.array([1, 2, 3]), 1, np.array([1, 2, 3])),
        (np.array([1, 2, 3]), 10, np.array([10, 20, 30])),
        (np.array([[1, 2], [3, 4]]), 1000, np.array([[1000, 2000], [3000, 4000]])),
        # NEGATIVE
        (np.array([1, 2, 3]), -1, np.array([1, 2, 3])),
        (np.array([10, 20, 30]), -10, np.array([1, 2, 3])),
        (np.array([[1000, 2000], [3000, 4000]]), -1000, np.array([[1, 2], [3, 4]])),
    ],
)
def test_apply_coordinate_scalar(data: NDArray, scalar: int, expected: NDArray) -> None:
    """Test applying coordinate scalar with negative and positive code."""
    result = _apply_coordinate_scalar(data, scalar)
    assert np.allclose(result, expected)
