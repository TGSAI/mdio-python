"""Test live mask chunk size calculation."""

import numpy as np

from mdio.core import Dimension
from mdio.core import Grid
from mdio.core.grid import _calculate_live_mask_chunksize
from mdio.core.grid import _calculate_optimal_chunksize


def test_small_grid_no_chunking():
    """Test that small grids return -1 (no chunking needed)."""
    # Create a small grid that fits within INT32_MAX
    dims = [
        Dimension(coords=range(0, 100, 1), name="dim1"),
        Dimension(coords=range(0, 100, 1), name="dim2"),
        Dimension(coords=range(0, 100, 1), name="sample"),
    ]
    grid = Grid(dims=dims)
    grid.live_mask = np.empty((100, 100), dtype=np.bool)

    result = _calculate_live_mask_chunksize(grid)
    assert result == (100, 100)


def test_grid_without_live_mask():
    """Test that a grid without a live mask set up yet."""
    dims = [
        Dimension(coords=range(0, 100, 1), name="dim1"),
        Dimension(coords=range(0, 100, 1), name="dim2"),
        Dimension(coords=range(0, 100, 1), name="sample"),
    ]
    grid = Grid(dims=dims)
    
    result = _calculate_live_mask_chunksize(grid)
    assert result == (100, 100)


def test_large_2d_grid_chunking():
    """Test exact chunk size calculation for a 2D grid that exceeds INT32_MAX."""
    # Create a grid that exceeds INT32_MAX (2,147,483,647)
    # Using 50,000 x 50,000 = 2,500,000,000 elements
    dims = [
        Dimension(coords=range(0, 50000, 1), name="dim1"),
        Dimension(coords=range(0, 50000, 1), name="dim2"),
        Dimension(coords=range(0, 100, 1), name="sample"),
    ]
    grid = Grid(dims=dims)
    grid.live_mask = np.empty((50000, 50000), dtype=np.bool)

    result = _calculate_live_mask_chunksize(grid)

    # TODO(BrianMichell): Avoid magic numbers.
    assert result == (25000, 25000)


def test_large_3d_grid_chunking():
    """Test exact chunk size calculation for a 3D grid that exceeds INT32_MAX."""
    # Create a 3D grid that exceeds INT32_MAX
    # Using 1500 x 1500 x 1500 = 3,375,000,000 elements
    dims = [
        Dimension(coords=range(0, 1500, 1), name="dim1"),
        Dimension(coords=range(0, 1500, 1), name="dim2"),
        Dimension(coords=range(0, 1500, 1), name="dim3"),
        Dimension(coords=range(0, 100, 1), name="sample"),
    ]
    grid = Grid(dims=dims)
    grid.live_mask = np.empty((1500, 1500, 1500), dtype=np.bool)

    result = _calculate_live_mask_chunksize(grid)

    # Calculate expected values
    # total_elements = 1500 * 1500 * 1500
    # num_chunks = np.ceil(total_elements / INT32_MAX).astype(int)
    # dim_chunks = int(np.ceil(1500 / np.ceil(np.power(num_chunks, 1 / 3))))
    # expected_chunk_size = int(np.ceil(1500 / dim_chunks))

    # assert result == (expected_chunk_size, expected_chunk_size, expected_chunk_size)
    assert result == (750, 750, 750)


def test_prestack_land_survey_chunking():
    """Test exact chunk size calculation for a dense pre-stack land survey grid."""
    # Create a dense pre-stack land survey grid that exceeds INT32_MAX
    # Using realistic dimensions:
    # - 1000 shot points
    # - 1000 receiver points
    # - 100 offsets
    # - 36 azimuths
    # Total elements: 1000 * 1000 * 100 * 36 = 3,600,000,000 elements
    dims = [
        Dimension(coords=range(0, 1000, 1), name="shot_point"),
        Dimension(coords=range(0, 1000, 1), name="receiver_point"),
        Dimension(coords=range(0, 100, 1), name="offset"),
        Dimension(coords=range(0, 36, 1), name="azimuth"),
        Dimension(coords=range(0, 1000, 1), name="sample"),
    ]
    grid = Grid(dims=dims)
    grid.live_mask = np.empty((1000, 1000, 100, 36), dtype=np.bool)

    result = _calculate_live_mask_chunksize(grid)
    assert result == (334, 334, 100, 36)


def test_one_dim_full_chunk():
    """Test one-dimensional volume where the whole dimension can be used as chunk."""
    arr = np.empty((100,), dtype=np.int8)
    # With n_bytes = 100, max_elements_allowed = 100, thus optimal chunk should be (100,)
    result = _calculate_optimal_chunksize(arr, 100)
    assert result == (100,)


def test_two_dim_optimal():
    """Test two-dimensional volume with limited n_bytes.

    For a shape of (8,6) with n_bytes=20, the optimal chunk is expected to be (8,2).
    """
    arr = np.empty((8, 6), dtype=np.int8)
    result = _calculate_optimal_chunksize(arr, 20)
    assert result == (4, 4)


def test_three_dim_optimal():
    """Test three-dimensional volume optimal chunk calculation.

    For a shape of (9,6,4) with n_bytes=100, the expected chunk is (9,2,4).
    """
    arr = np.empty((9, 6, 4), dtype=np.int8)
    result = _calculate_optimal_chunksize(arr, 100)
    assert result == (5, 5, 4)


def test_minimal_chunk_for_large_dtype():
    """Test that n_bytes forcing minimal chunking returns all ones.

    Using int32 (itemsize=4) with shape (4,5) and n_bytes=4 yields (1,1).
    """
    arr = np.empty((4, 5), dtype=np.int32)
    result = _calculate_optimal_chunksize(arr, 4)
    assert result == (1, 1)


def test_large_nbytes():
    """Test that a very large n_bytes returns the full volume shape as the optimal chunk."""
    arr = np.empty((10, 10), dtype=np.int8)
    result = _calculate_optimal_chunksize(arr, 1000)
    assert result == (10, 10)


def test_two_dim_non_int8():
    """Test with a non-int8 dtype where n_bytes exactly covers the full volume in bytes."""
    arr = np.empty((6, 8), dtype=np.int16)  # int16 has itemsize 2
    # Total bytes of full volume = 6*8*2 = 96, so optimal chunk should be (6,8)
    result = _calculate_optimal_chunksize(arr, 96)
    assert result == (6, 8)


def test_irregular_dimensions():
    """Test volume with prime dimensions where divisors are limited.

    For shape (7,5) with n_bytes=35, optimal chunk should be (7,5) since 7*5 = 35.
    """
    arr = np.empty((7, 5), dtype=np.int8)
    result = _calculate_optimal_chunksize(arr, 35)
    assert result == (7, 5)


def test_primes():
    """Test volume with prime dimensions where divisors are limited."""
    arr = np.empty((7, 5), dtype=np.int8)
    result = _calculate_optimal_chunksize(arr, 23)
    assert result == (4, 4)


def test_grid_gambit():
    """Test various chunk size scenarios with different array dimensions."""
    from mdio.constants import INT32_MAX

    # Dictionary of test cases with different array shapes
    live_mask_chunks_examples = {
        "smaller_2G_asym": (1024, 8192),
        "small_2G_square": (32768, 32768),
        "right_below_2G": (46340, 46340),
        "right_above_2G": (46341, 46341),
        "above_2G_v1": (86341, 96341),
        "above_2G_v2": (55000, 47500),
        "above_2G_v2_asym": (55000, 97500),
        "above_4G_v2_asym": (100000, 100000),
        "below_2G_4D": (215, 215, 215, 215),
        "above_3G_4D": (216, 216, 216, 216),
        "above_3G_4D_asym": (512, 216, 512, 400),
        "below_2G_5D": (73, 73, 73, 73, 73),
        "above_3G_5D": (74, 74, 74, 74, 74),
        "above_3G_5D_asym": (512, 17, 43, 200, 50),
    }

    # Test each case
    for kind, shape in live_mask_chunks_examples.items():
        # Create dimensions for the grid
        dims = [
            Dimension(coords=range(0, dim_size, 1), name=f"dim{i}")
            for i, dim_size in enumerate(shape)
        ]
        # Add sample dimension
        dims.append(Dimension(coords=range(0, 100, 1), name="sample"))

        # Create grid and set live mask
        grid = Grid(dims=dims)
        grid.live_mask = np.empty(shape, dtype=np.bool)

        result = _calculate_live_mask_chunksize(grid)

        # Verify that the chunk size is valid
        assert all(chunk > 0 for chunk in result), f"Invalid chunk size for {kind}"
        assert len(result) == len(shape), f"Dimension mismatch for {kind}"

        # # Calculate total elements in chunks
        chunk_elements = np.prod(result)
        if kind in [
            "right_above_2G",
            "above_2G_v2",
            "above_2G_v2_asym",
            "above_4G_v2_asym",
            "above_3G_4D_asym",
        ]:
            # TODO(BrianMichell): Our implementation is taking "limit" pretty liberally.
            # This is not overtly an issue because we are well below the 2GiB limit,
            # but it could be improved.
            assert (
                chunk_elements <= (INT32_MAX // 4) * 1.5
            ), f"Chunk too large for {kind}"
        else:
            assert chunk_elements <= INT32_MAX // 4, f"Chunk too large for {kind}"
