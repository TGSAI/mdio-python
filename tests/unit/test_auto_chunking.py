"""Test live mask chunk size calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from mdio.core.utils_write import MAX_SIZE_LIVE_MASK
from mdio.core.utils_write import get_constrained_chunksize
from mdio.core.utils_write import get_live_mask_chunksize

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


@pytest.mark.parametrize(
    ("shape", "dtype", "limit", "expected_chunks"),
    [
        ((100,), "int8", 100, (100,)),  # 1D full chunk
        ((8, 6), "int8", 20, (4, 4)),  # 2D adjusted int8
        ((6, 8), "int16", 96, (6, 8)),  # 2D small int16
        ((9, 6, 4), "int8", 100, (5, 5, 4)),  # 3D adjusted
        ((4, 5), "int32", 4, (1, 1)),  # test minimum edge case
        ((10, 10), "int8", 1000, (10, 10)),  # big limit
        ((7, 5), "int8", 35, (7, 5)),  # test full primes
        ((7, 5), "int8", 23, (4, 4)),  # test adjusted primes
    ],
)
@pytest.mark.filterwarnings("ignore:chunk size balancing not possible:UserWarning")
def test_auto_chunking(
    shape: tuple[int, ...],
    dtype: DTypeLike,
    limit: int,
    expected_chunks: tuple[int, ...],
) -> None:
    """Test automatic chunking based on size limit and an array spec."""
    result = get_constrained_chunksize(shape, dtype, limit)
    assert result == expected_chunks


class TestAutoChunkLiveMask:
    """Test class for live mask auto chunking."""

    @pytest.mark.parametrize(
        ("shape", "expected_chunks"),
        [
            ((100,), (100,)),  # small 1d
            ((100, 100), (100, 100)),  # small 2d
            ((50000, 50000), (16667, 16667)),  # large 2d
            ((1500, 1500, 1500), (750, 750, 750)),  # large 3d
            ((1000, 1000, 100, 36), (250, 250, 100, 36)),  # large 4d
        ],
    )
    def test_auto_chunk_live_mask(
        self,
        shape: tuple[int, ...],
        expected_chunks: tuple[int, ...],
    ) -> None:
        """Test auto chunked live mask is within expected number of bytes."""
        result = get_live_mask_chunksize(shape)
        assert result == expected_chunks

    @pytest.mark.parametrize(
        "shape",
        [
            # Below are >250MiB. Smaller ones tested above
            (32768, 32768),
            (46341, 46341),
            (86341, 96341),
            (55000, 97500),
            (100000, 100000),
            (512, 216, 512, 400),
            (64, 128, 64, 32, 64),
            (512, 17, 43, 200, 50),
        ],
    )
    @pytest.mark.filterwarnings("ignore:chunk size balancing not possible:UserWarning")
    def test_auto_chunk_live_mask_nbytes(self, shape: tuple[int, ...]) -> None:
        """Test auto chunked live mask is within expected number of bytes."""
        result = get_live_mask_chunksize(shape)
        chunk_elements = np.prod(result)

        # We want them to be 250MB +/- 50%
        assert chunk_elements > MAX_SIZE_LIVE_MASK * 0.75
        assert chunk_elements < MAX_SIZE_LIVE_MASK * 1.25
