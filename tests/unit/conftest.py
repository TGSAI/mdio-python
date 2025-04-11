"""Extra configurations for unit tests."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray
from zarr import consolidate_metadata
from zarr import open_group

from mdio import MDIOReader
from mdio import MDIOWriter
from mdio.core import Dimension
from mdio.core import Grid
from mdio.core.factory import MDIOCreateConfig
from mdio.core.factory import MDIOVariableConfig
from mdio.core.factory import create_empty
from mdio.core.utils_write import write_attribute


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
def mock_mdio_path(tmp_path_factory):
    """Make a mocked MDIO store for writing."""
    return tmp_path_factory.mktemp("mdio") / "mock.mdio"


@pytest.fixture
def mock_ilxl_values(mock_grid: Grid) -> tuple[NDArray, ...]:
    """Make some fake X/Y coordinates."""
    il_coords = mock_grid.select_dim("inline").coords
    xl_coords = mock_grid.select_dim("crossline").coords
    xl_grid, il_grid = np.meshgrid(xl_coords, il_coords)
    return il_grid, xl_grid


@pytest.fixture
def mock_bin():
    """Make a mock binary header."""
    return dict(bin_hdr1=5, bin_hdr2=10)


@pytest.fixture
def mock_data(mock_grid: Grid, mock_ilxl_values: tuple[NDArray, ...]) -> NDArray:
    """Make some mock data as numpy array."""
    il_grid, xl_grid = mock_ilxl_values
    sample_axis = mock_grid.select_dim("sample").coords
    data = il_grid / xl_grid
    data = data[..., None] + sample_axis[None, None, :]

    return data


@pytest.fixture
def mock_mdio(
    mock_mdio_path: Path,
    mock_dimensions: list[Dimension],
    mock_coords: tuple[NDArray],
    mock_data: NDArray,
    mock_bin: dict[str, int],
):
    """This mocks most of mdio.converters.segy in memory."""
    zarr.config.set({"default_zarr_format": 2, "write_empty_chunks": False})
    zarr_root = open_group(mock_mdio_path, mode="w")

    create_zarr_hierarchy(
        root_group=zarr_root,
        overwrite=True,
    )

    data_grp = zarr_root["data"]
    metadata_grp = zarr_root["metadata"]

    dimensions_dict = [dim.to_dict() for dim in mock_dimensions]
    grid = Grid(mock_dimensions)

    il_grid, xl_grid = mock_coords
    mock_inline, mock_crossline = il_grid.ravel(), xl_grid.ravel()

    mock_dtype = np.dtype([("inline", "i4"), ("crossline", "i4")])
    mock_headers = np.empty(il_grid.size, dtype=mock_dtype)
    mock_headers["inline"] = mock_inline
    mock_headers["crossline"] = mock_crossline

    grid.build_map(mock_headers)

    trace_count = np.count_nonzero(grid.live_mask)
    write_attribute(name="dimension", zarr_group=zarr_root, attribute=dimensions_dict)
    il_grid, xl_grid = mock_ilxl_values
    mock_header_dtype = np.dtype([("inline", "i4"), ("crossline", "i4")])
    mock_grid.live_mask = np.ones(mock_grid.shape[:-1], dtype=bool)

    var = MDIOVariableConfig(
        name="chunked_012",
        dtype="float64",
        chunks=mock_grid.shape,
        header_dtype=mock_header_dtype,
    )

    conf = MDIOCreateConfig(path=mock_mdio_dir, grid=mock_grid, variables=[var])
    zarr_root = create_empty(conf, overwrite=True)
    trace_count = np.count_nonzero(mock_grid.live_mask)
    write_attribute(name="trace_count", zarr_group=zarr_root, attribute=trace_count)

    writer = MDIOWriter(mock_mdio_dir)
    writer.binary_header = mock_bin

    writer._headers["inline"] = il_grid
    writer._headers["crossline"] = xl_grid
    writer[:] = mock_data

    stats = {
        "mean": mock_data.mean(),
        "std": mock_data.std(),
        "rms": np.sqrt((mock_data**2).sum() / mock_data.size),
        "min": mock_data.min(),
        "max": mock_data.max(),
    }
    writer.stats = stats
    return zarr_root


@pytest.fixture
def mock_reader(mock_mdio: Path) -> MDIOReader:
    """Reader that points to the mocked data to be used later."""
    return MDIOReader(mock_mdio.__str__())


@pytest.fixture
def mock_reader_cached(mock_mdio: Path) -> MDIOReader:
    """Reader that points to the mocked data to be used later. (with local caching)."""
    return MDIOReader(
        mock_mdio.__str__(),
        disk_cache=True,
        storage_options={"simplecache": {"cache_storage": "./mdio_test_cache"}},
    )
