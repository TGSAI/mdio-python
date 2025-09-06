"""Extra configurations for unit tests."""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mdio.core import Dimension
from mdio.core import Grid

if TYPE_CHECKING:
    from numpy.typing import NDArray

API_VERSION = metadata.version("multidimio")

TEST_DIMS = [
    Dimension(name="inline", coords=np.arange(101, 131, 2)),
    Dimension(name="crossline", coords=np.arange(10, 20, 1)),
    Dimension(name="sample", coords=np.arange(0, 100, 5)),
]


@pytest.fixture
def mock_grid() -> Grid:
    """Make a mock grid using test dimensions."""
    return Grid(dims=TEST_DIMS)


@pytest.fixture(scope="module")
def mock_mdio_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Make a mocked MDIO dir for writing."""
    return str(tmp_path_factory.mktemp("mdio"))


@pytest.fixture
def mock_ilxl_values(mock_grid: Grid) -> tuple[NDArray, ...]:
    """Make some fake X/Y coordinates."""
    il_coords = mock_grid.select_dim("inline").coords
    xl_coords = mock_grid.select_dim("crossline").coords
    xl_grid, il_grid = np.meshgrid(xl_coords, il_coords)
    return il_grid, xl_grid


@pytest.fixture
def mock_bin() -> dict[str, int]:
    """Make a mock binary header."""
    return {"bin_hdr1": 5, "bin_hdr2": 10}


@pytest.fixture
def mock_data(mock_grid: Grid, mock_ilxl_values: tuple[NDArray, ...]) -> NDArray:
    """Make some mock data as numpy array."""
    il_grid, xl_grid = mock_ilxl_values
    sample_axis = mock_grid.select_dim("sample").coords
    data = il_grid / xl_grid
    return data[..., None] + sample_axis[None, None, :]
