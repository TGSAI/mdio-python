"""Unit tests for the type converter module."""

import numpy as np
from xarray import DataArray as xr_DataArray
from xarray import Dataset as xr_Dataset

from mdio.core.indexing import ChunkIterator


def test_chunk_iterator_returning_dict() -> None:
    """Test the ChunkIterator class."""
    dims = ["inline", "crossline", "depth"]
    chunks = (3, 4, 5)

    shape = (6, 12, 20)
    iter1 = ChunkIterator(shape=shape, chunks=chunks, dim_names=dims)
    assert iter1.arr_shape == shape
    assert iter1.dims == dims
    assert iter1.len_chunks == chunks
    assert iter1.dim_chunks == (2, 3, 4)
    assert iter1.num_chunks == 24

    shape = (5, 11, 19)
    iter2 = ChunkIterator(shape=shape, chunks=chunks, dim_names=dims)
    assert iter2.dim_chunks == (2, 3, 4)
    assert iter2.num_chunks == 24

    # Its purpose is to confirm that all slices are created of the same size,
    # even if the last slice should have been smaller.
    for _ in range(13):  # element index 12
        region = iter1.__next__()
    assert region == {
        "inline": slice(3, 6, None),
        "crossline": slice(0, 4, None),
        "depth": slice(0, 5, None),
    }

    for _ in range(13):  # element index 12
        region = iter2.__next__()
    assert region == {
        "inline": slice(3, 6, None),
        "crossline": slice(0, 4, None),
        "depth": slice(0, 5, None),
    }


def test_chunk_iterator_returning_tuple() -> None:
    """Test the ChunkIterator class."""
    chunks = (3, 4, 5)

    shape = (6, 12, 20)
    iter1 = ChunkIterator(shape=shape, chunks=chunks)
    assert iter1.arr_shape == shape
    assert iter1.dims is None
    assert iter1.len_chunks == chunks
    assert iter1.dim_chunks == (2, 3, 4)
    assert iter1.num_chunks == 24

    shape = (5, 11, 19)
    iter2 = ChunkIterator(shape=shape, chunks=chunks)
    assert iter2.dim_chunks == (2, 3, 4)
    assert iter2.num_chunks == 24

    # Its purpose is to confirm that all slices are created of the same size,
    # even if the last slice should have been smaller.
    for _ in range(13):  # element index 12
        region = iter1.__next__()
    assert region == (slice(3, 6, None), slice(0, 4, None), slice(0, 5, None))

    for _ in range(13):  # element index 12
        region = iter2.__next__()
    assert region == (slice(3, 6, None), slice(0, 4, None), slice(0, 5, None))


def val(shape: tuple[int, int, int], i: int, j: int, k: int) -> int:
    """Calculate the linear index in a 3D array."""
    return i * (shape[1] * shape[2]) + j * shape[2] + k


def mock_trace_worker(
    shape: tuple[int, int, int], region: dict[str, slice], dataset: xr_Dataset, grid_map: np.array
) -> None:
    """Mock trace worker function.

    Note:
        Xarray, Zarr, and NumPy automatically truncates the slice to the valid bounds of the array
        (see the test above, where the last chunk is always of the same size)
        and does not raise an error. However, if one attempts to access an element at an index
        that is out of bounds, you will get an IndexError
    """
    # We used a 2D selection with 2D index_slices
    assert grid_map.shape == (3, 4, 20)
    # We used a 3D selection with isel()
    assert tuple(dataset.sizes[d] for d in region) == (3, 4, 5)

    dimension_names = list(dataset.sizes)

    slice0 = region[dimension_names[0]]
    slice1 = region[dimension_names[1]]
    slice2 = region[dimension_names[2]]
    for ii, i in enumerate(range(slice0.start, min(slice0.stop, shape[0]))):
        for jj, j in enumerate(range(slice1.start, min(slice1.stop, shape[1]))):
            for kk, k in enumerate(range(slice2.start, min(slice2.stop, shape[2]))):
                # Validate that we've got the sample indexing right
                assert dataset["amplitude"].values[ii, jj, kk] == val(shape, i, j, k)
                # NOTE: grid_map is 2D, so we need to use k for the depth dimension
                assert dataset["amplitude"].values[ii, jj, kk] == grid_map[ii, jj, k]


def test_chunk_iterator_with_dataset() -> None:
    """Test the ChunkIterator with a dataset."""
    shape = (6, 12, 20)
    dims = ["inline", "crossline", "depth"]
    chunks = (3, 4, 5)

    data3 = np.arange(shape[0] * shape[1] * shape[2]).reshape(shape)
    amplitude = xr_DataArray(data3, dims=dims, name="amplitude")
    ds = xr_Dataset({"amplitude": amplitude})

    chunk_iter = ChunkIterator(shape, chunks, dims)
    for region in chunk_iter:
        # If one needs both a dict and a tuple of slices,
        # one can use the following line an example to strip dim names out
        index_slices = tuple(region[key] for key in dims[:-1])
        # The .isel() method takes keyword arguments, region, where each keyword corresponds
        # to a dimension name and the value is an integer, a slice object (our case),
        # or an array-like object
        mock_trace_worker(shape, region, ds.isel(region), amplitude[index_slices])
