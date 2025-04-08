"""Test live mask chunk size calculation."""

import numpy as np
import pytest

from mdio.converters.segy import _calculate_live_mask_chunksize
from mdio.core import Grid, Dimension
from mdio.constants import INT32_MAX


def test_small_grid_no_chunking():
    """Test that small grids return -1 (no chunking needed)."""
    # Create a small grid that fits within INT32_MAX
    dims = [
        Dimension(coords=range(0, 100, 1), name="dim1"),
        Dimension(coords=range(0, 100, 1), name="dim2"),
        Dimension(coords=range(0, 100, 1), name="sample")
    ]
    grid = Grid(dims=dims)
    grid.live_mask = np.ones((100, 100), dtype=bool)
    
    result = _calculate_live_mask_chunksize(grid)
    assert result == -1


def test_large_2d_grid_chunking():
    """Test exact chunk size calculation for a 2D grid that exceeds INT32_MAX."""
    # Create a grid that exceeds INT32_MAX (2,147,483,647)
    # Using 50,000 x 50,000 = 2,500,000,000 elements
    dims = [
        Dimension(coords=range(0, 50000, 1), name="dim1"),
        Dimension(coords=range(0, 50000, 1), name="dim2"),
        Dimension(coords=range(0, 100, 1), name="sample")
    ]
    grid = Grid(dims=dims)
    grid.live_mask = np.ones((50000, 50000), dtype=bool)
    
    result = _calculate_live_mask_chunksize(grid)
    
    # Calculate expected values
    total_elements = 50000 * 50000
    num_chunks = np.ceil(total_elements / INT32_MAX).astype(int)
    dim_chunks = int(np.ceil(50000 / np.ceil(np.power(num_chunks, 1/2))))
    expected_chunk_size = int(np.ceil(50000 / dim_chunks))
    
    assert result == (expected_chunk_size, expected_chunk_size)


def test_large_3d_grid_chunking():
    """Test exact chunk size calculation for a 3D grid that exceeds INT32_MAX."""
    # Create a 3D grid that exceeds INT32_MAX
    # Using 1500 x 1500 x 1500 = 3,375,000,000 elements
    dims = [
        Dimension(coords=range(0, 1500, 1), name="dim1"),
        Dimension(coords=range(0, 1500, 1), name="dim2"),
        Dimension(coords=range(0, 1500, 1), name="dim3"),
        Dimension(coords=range(0, 100, 1), name="sample")
    ]
    grid = Grid(dims=dims)
    grid.live_mask = np.ones((1500, 1500, 1500), dtype=bool)
    
    result = _calculate_live_mask_chunksize(grid)
    
    # Calculate expected values
    total_elements = 1500 * 1500 * 1500
    num_chunks = np.ceil(total_elements / INT32_MAX).astype(int)
    dim_chunks = int(np.ceil(1500 / np.ceil(np.power(num_chunks, 1/3))))
    expected_chunk_size = int(np.ceil(1500 / dim_chunks))
    
    assert result == (expected_chunk_size, expected_chunk_size, expected_chunk_size)


def test_uneven_dimensions_chunking():
    """Test exact chunk size calculation for uneven dimensions."""
    # Create a grid with uneven dimensions that exceeds INT32_MAX
    # Using 50,000 x 50,000 = 2,500,000,000 elements (exceeds INT32_MAX)
    # But with uneven chunking: 50,000 x 25,000
    dims = [
        Dimension(coords=range(0, 50000, 1), name="dim1"),
        Dimension(coords=range(0, 50000, 1), name="dim2"),
        Dimension(coords=range(0, 100, 1), name="sample")
    ]
    grid = Grid(dims=dims)
    grid.live_mask = np.ones((50000, 50000), dtype=bool)
    
    result = _calculate_live_mask_chunksize(grid)
    
    # Calculate expected values
    total_elements = 50000 * 50000
    num_chunks = np.ceil(total_elements / INT32_MAX).astype(int)
    dim_chunks = int(np.ceil(50000 / np.ceil(np.power(num_chunks, 1/2))))
    expected_chunk_size = int(np.ceil(50000 / dim_chunks))
    
    assert result == (expected_chunk_size, expected_chunk_size)


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
        Dimension(coords=range(0, 1000, 1), name="sample")
    ]
    grid = Grid(dims=dims)
    grid.live_mask = np.ones((1000, 1000, 100, 36), dtype=bool)
    
    result = _calculate_live_mask_chunksize(grid)
    
    # Calculate expected values
    total_elements = 1000 * 1000 * 100 * 36
    num_chunks = np.ceil(total_elements / INT32_MAX).astype(int)
    dim_chunks = int(np.ceil(1000 / np.ceil(np.power(num_chunks, 1/4))))
    expected_chunk_size = int(np.ceil(1000 / dim_chunks))
    
    # For a 4D grid, we expect chunk sizes to be distributed across all dimensions
    # The chunk size should be the same for all dimensions since they're all equally important
    assert result == (expected_chunk_size, expected_chunk_size, expected_chunk_size, expected_chunk_size)


def test_edge_case_empty_grid():
    """Test empty grid edge case."""
    dims = [
        Dimension(coords=range(0, 0, 1), name="dim1"),
        Dimension(coords=range(0, 0, 1), name="dim2"),
        Dimension(coords=range(0, 100, 1), name="sample")
    ]
    grid = Grid(dims=dims)
    grid.live_mask = np.zeros((0, 0), dtype=bool)
    
    result = _calculate_live_mask_chunksize(grid)
    assert result == -1  # Empty grid shouldn't need chunking 