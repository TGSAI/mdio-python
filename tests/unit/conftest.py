"""Extra configurations for unit tests."""

from __future__ import annotations

from datetime import datetime
from importlib import metadata

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray
from zarr import Group
from zarr import consolidate_metadata
from zarr import open_group

from mdio import MDIOReader
from mdio.core import Dimension
from mdio.core import Grid
from mdio.core.utils_write import write_attribute
from mdio.segy.helpers_segy import create_zarr_hierarchy


API_VERSION = metadata.version("multidimio")

TEST_DIMS = {
    "inline": np.arange(101, 131, 2),
    "crossline": np.arange(10, 20, 1),
    "sample": np.arange(0, 100, 5),
}


@pytest.fixture(scope="module")
def mock_root_group(tmp_path_factory) -> Group:
    """Make a mocked MDIO store for writing."""
    zarr.config.set({"default_zarr_format": 2, "write_empty_chunks": False})
    tmp_dir = tmp_path_factory.mktemp("mdio")
    return open_group(tmp_dir.name, mode="w")


@pytest.fixture
def mock_dimensions():
    """Make some fake dimensions."""
    dimensions = [Dimension(coords, name) for name, coords in TEST_DIMS.items()]
    return dimensions


@pytest.fixture
def mock_coords():
    """Make some fake X/Y coordinates."""
    xl_grid, il_grid = np.meshgrid(TEST_DIMS["crossline"], TEST_DIMS["inline"])
    return il_grid, xl_grid


@pytest.fixture
def mock_text():
    """Make a mock text header."""
    return [f"{idx:02d} ab " * 16 for idx in range(40)]


@pytest.fixture
def mock_bin():
    """Make a mock binary header."""
    return dict(bin_hdr1=5, bin_hdr2=10)


@pytest.fixture
def mock_data(mock_coords):
    """Make some mock data as numpy array."""
    il_grid, xl_grid = mock_coords
    data = il_grid / xl_grid
    data = data[..., None] + TEST_DIMS["sample"][None, None, :]

    return data


@pytest.fixture
def mock_mdio(
    mock_root_group: Group,
    mock_dimensions: list[Dimension],
    mock_coords: tuple[NDArray],
    mock_data: NDArray,
    mock_text: list[str],
    mock_bin: dict[str, int],
):
    """This mocks most of mdio.converters.segy in memory."""
    zarr_root = create_zarr_hierarchy(
        root_group=mock_root_group,
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

    live_mask_arr = zarr_root["metadata"].create_array(
        name="live_mask",
        dtype="bool",
        shape=grid.shape[:-1],
        chunks=grid.shape[:-1],
        chunk_key_encoding={"name": "v2", "separator": "/"},
    )
    live_mask_arr[...] = grid.live_mask[...]

    write_attribute(name="created", zarr_group=zarr_root, attribute=str(datetime.now()))
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

    data_arr = data_grp.create_array(
        "chunked_012",
        data=mock_data,
        chunk_key_encoding={"name": "v2", "separator": "/"},
    )

    metadata_grp.create_array(
        data=il_grid * xl_grid,
        name="_".join(["chunked_012", "trace_headers"]),
        shape=grid.shape[:-1],  # Same spatial shape as data
        chunks=data_arr.chunks[:-1],  # Same spatial chunks as data
        chunk_key_encoding={"name": "v2", "separator": "/"},
    )

    consolidate_metadata(mock_root_group.store)

    return zarr_root


@pytest.fixture
def mock_reader(mock_mdio: Group) -> MDIOReader:
    """Reader that points to the mocked data to be used later."""
    return MDIOReader(mock_mdio.store.path)


@pytest.fixture
def mock_reader_cached(mock_mdio: Group) -> MDIOReader:
    """Reader that points to the mocked data to be used later. (with local caching)."""
    return MDIOReader(
        mock_mdio.store.path,
        disk_cache=True,
        storage_options={"simplecache": {"cache_storage": "./mdio_test_cache"}},
    )
