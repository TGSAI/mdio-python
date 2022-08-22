"""Extra configurations for unit tests."""


from datetime import datetime
from importlib import metadata

import numpy as np
import pytest
import zarr

from mdio import MDIOReader
from mdio.core import Dimension
from mdio.core import Grid
from mdio.core.utils_write import write_attribute
from mdio.segy.helpers_segy import create_zarr_hierarchy


API_VERSION = metadata.version("mdio")

TEST_DIMS = {
    "inline": np.arange(101, 131, 2),
    "crossline": np.arange(10, 20, 1),
    "sample": np.arange(0, 100, 5),
}


@pytest.fixture(scope="module")
def mock_store(tmp_path_factory):
    """Make a mocked MDIO store for writing."""
    tmp_dir = tmp_path_factory.mktemp("mdio")
    return zarr.storage.FSStore(tmp_dir.name, dimension_separator="/")


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
def mock_mdio(mock_store, mock_dimensions, mock_coords, mock_data, mock_text, mock_bin):
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
    mock_headers = il_grid.ravel(), xl_grid.ravel()
    grid.build_map(np.column_stack(mock_headers))

    trace_count = np.count_nonzero(grid.live_mask)
    write_attribute(name="dimension", zarr_group=zarr_root, attribute=dimensions_dict)
    write_attribute(name="trace_count", zarr_group=zarr_root, attribute=trace_count)

    zarr_root["metadata"].create_dataset(
        data=grid.live_mask,
        name="live_mask",
        shape=grid.shape[:-1],
        chunks=-1,
    )

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

    data_arr = data_grp.create_dataset("chunked_012", data=mock_data)

    metadata_grp.create_dataset(
        data=il_grid * xl_grid,
        name="_".join(["chunked_012", "trace_headers"]),
        shape=grid.shape[:-1],  # Same spatial shape as data
        chunks=data_arr.chunks[:-1],  # Same spatial chunks as data
    )

    zarr.consolidate_metadata(mock_store)

    return zarr_root


@pytest.fixture
def mock_reader(mock_mdio, mock_store):
    """Reader that points to the mocked data to be used later."""
    return MDIOReader(mock_store)
