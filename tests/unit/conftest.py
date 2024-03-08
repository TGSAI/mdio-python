"""Extra configurations for unit tests."""

from __future__ import annotations

from datetime import datetime
from importlib import metadata
from typing import TYPE_CHECKING

import numpy as np
import pytest
from zarr import Group
from zarr import consolidate_metadata
from zarr.storage import FSStore

from mdio import MDIOReader
from mdio.core import Dimension
from mdio.core import Grid
from mdio.core.utils_write import write_attribute
from mdio.seismic.helpers_segy import create_zarr_hierarchy


if TYPE_CHECKING:
    from numpy.typing import NDArray

API_VERSION = metadata.version("multidimio")

TEST_DIMS = {
    "inline": np.arange(101, 131, 2),
    "crossline": np.arange(10, 20, 1),
    "sample": np.arange(0, 100, 5),
}


@pytest.fixture(scope="module")
def mock_store(tmp_path_factory: pytest.TempPathFactory) -> FSStore:
    """Make a mocked MDIO store for writing."""
    tmp_dir = tmp_path_factory.mktemp("mdio")
    return FSStore(tmp_dir.name)


@pytest.fixture()
def mock_dimensions() -> list[Dimension]:
    """Make some fake dimensions."""
    return [Dimension(coords, name) for name, coords in TEST_DIMS.items()]


@pytest.fixture()
def mock_coords() -> tuple[NDArray[int], NDArray[int]]:
    """Make some fake X/Y coordinates."""
    xl_grid, il_grid = np.meshgrid(TEST_DIMS["crossline"], TEST_DIMS["inline"])
    return il_grid, xl_grid


@pytest.fixture()
def mock_text() -> list[str]:
    """Make a mock text header."""
    return [f"{idx:02d} ab " * 16 for idx in range(40)]


@pytest.fixture()
def mock_bin() -> dict[str, int]:
    """Make a mock binary header."""
    return {"bin_hdr1": 5, "bin_hdr2": 10}


@pytest.fixture()
def mock_data(mock_coords: tuple[NDArray[int], NDArray[int]]) -> NDArray[float]:
    """Make some mock data as numpy array."""
    il_grid, xl_grid = mock_coords
    data = il_grid / xl_grid

    return data[..., None] + TEST_DIMS["sample"][None, None, :]


@pytest.fixture()
def mock_mdio(  # noqa: PLR0913
    mock_store: FSStore,
    mock_dimensions: list[Dimension],
    mock_coords: tuple[NDArray],
    mock_data: NDArray,
    mock_text: list[str],
    mock_bin: dict[str, int],
) -> Group:
    """This mocks most of mdio.converters.segy in memory."""
    zarr_root = create_zarr_hierarchy(
        store=mock_store,
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
    write_attribute(name="trace_count", zarr_group=zarr_root, attribute=trace_count)

    zarr_root["metadata"].create_dataset(
        data=grid.live_mask,
        name="live_mask",
        shape=grid.shape[:-1],
        chunks=-1,
        dimension_separator="/",
    )

    iso_time = str(datetime.now().astimezone().isoformat())

    write_attribute(name="created", zarr_group=zarr_root, attribute=iso_time)
    write_attribute(name="api_version", zarr_group=zarr_root, attribute=API_VERSION)
    write_attribute(name="text_header", zarr_group=metadata_grp, attribute=mock_text)
    write_attribute(name="binary_header", zarr_group=metadata_grp, attribute=mock_bin)

    stats = {
        "mean": mock_data.mean(),
        "std": mock_data.std(),
        "rms": np.sqrt((mock_data**2).sum() / mock_data.size),
        "min": mock_data.min(),
        "max": mock_data.max(),
    }

    for key, value in stats.items():
        write_attribute(name=key, zarr_group=zarr_root, attribute=value)

    data_arr = data_grp.create_dataset(
        "chunked_012",
        data=mock_data,
        dimension_separator="/",
    )

    metadata_grp.create_dataset(
        data=il_grid * xl_grid,
        name="chunked_012_trace_headers",
        shape=grid.shape[:-1],  # Same spatial shape as data
        chunks=data_arr.chunks[:-1],  # Same spatial chunks as data
        dimension_separator="/",
    )

    consolidate_metadata(mock_store)

    return zarr_root


@pytest.fixture()
def mock_reader(mock_mdio: Group) -> MDIOReader:
    """Reader that points to the mocked data to be used later."""
    return MDIOReader(mock_mdio.store.path)


@pytest.fixture()
def mock_reader_cached(mock_mdio: Group) -> MDIOReader:
    """Reader that points to the mocked data to be used later. (with local caching)."""
    return MDIOReader(
        mock_mdio.store.path,
        disk_cache=True,
        storage_options={"simplecache": {"cache_storage": "./mdio_test_cache"}},
    )
