"""Module for testing NumPy to MDIO conversion functionality.

This module contains tests for the `numpy_to_mdio` function, ensuring proper conversion
of NumPy arrays to MDIO format, including validation of grid dimensions, chunk sizes,
and coordinate handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.testing as npt
import pytest

from mdio.api.accessor import MDIOReader
from mdio.converters.numpy import numpy_to_mdio
from mdio.core.dimension import Dimension
from mdio.core.grid import Grid


if TYPE_CHECKING:
    from numpy.typing import NDArray


TEST_DIMS = [
    Dimension(name="inline", coords=np.arange(101, 131, 2)),
    Dimension(name="crossline", coords=np.arange(10, 20, 1)),
    Dimension(name="sample", coords=np.arange(0, 100, 5)),
]


@pytest.fixture
def mock_grid() -> Grid:
    """Make a mock grid using test dimensions."""
    return Grid(dims=TEST_DIMS)


@pytest.fixture
def mock_array(mock_grid: Grid) -> NDArray:
    """Make a mock array using mock grid."""
    rng = np.random.default_rng()
    return rng.uniform(size=mock_grid.shape).astype("float32")


CHUNK_SIZE = (8, 8, 8)


def test_npy_to_mdio(mock_array: NDArray, mock_grid: Grid) -> None:
    """Test basic NumPy to MDIO conversion without custom coordinates."""
    numpy_to_mdio(mock_array, "memory://npy.mdio", CHUNK_SIZE)
    reader = MDIOReader("memory://npy.mdio")

    npt.assert_array_equal(reader._traces, mock_array)
    assert reader.grid.dim_names == ("dim_0", "dim_1", "sample")
    assert reader.chunks == CHUNK_SIZE
    assert reader.shape == mock_grid.shape
    assert reader.grid.dims != mock_grid.dims


def test_npy_to_mdio_coords(mock_array: NDArray, mock_grid: Grid) -> None:
    """Test NumPy to MDIO conversion with custom coordinates."""
    index_names = mock_grid.dim_names
    index_coords = {dim.name: dim.coords for dim in mock_grid.dims}
    numpy_to_mdio(
        mock_array, "memory://npy_coord.mdio", CHUNK_SIZE, index_names, index_coords
    )
    reader = MDIOReader("memory://npy_coord.mdio")

    npt.assert_array_equal(reader._traces, mock_array)
    assert reader.chunks == CHUNK_SIZE
    assert reader.shape == mock_grid.shape
    assert reader.grid.dims == mock_grid.dims


def test_npy_to_mdio_chunksize_mismatch(mock_array: NDArray, mock_grid: Grid) -> None:
    """Test error handling for mismatched chunk size dimensions."""
    with pytest.raises(ValueError, match="equal to array dimensions"):
        numpy_to_mdio(mock_array, "", (5, 10, 15, 20, 25))


def test_npy_to_mdio_coord_missing(mock_array: NDArray, mock_grid: Grid) -> None:
    """Test error handling for missing coordinate names."""
    index_names = ["mismatch", "dimension", "names"]
    index_coords = {dim.name: dim.coords for dim in mock_grid.dims}

    with pytest.raises(ValueError, match="not found in index_coords"):
        numpy_to_mdio(
            mock_array,
            "",
            CHUNK_SIZE,
            index_names,
            index_coords,
        )


def test_npy_to_mdio_coord_size_error(mock_array: NDArray, mock_grid: Grid) -> None:
    """Test error handling for coordinate size mismatch."""
    index_names = mock_grid.dim_names
    index_coords = {dim.name: np.arange(5) for dim in mock_grid.dims}

    with pytest.raises(ValueError, match="does not match array dimension"):
        numpy_to_mdio(
            mock_array,
            "",
            CHUNK_SIZE,
            index_names,
            index_coords,
        )
