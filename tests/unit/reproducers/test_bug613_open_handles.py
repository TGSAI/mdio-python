"""zarr: 3.1.2 Reproducer for bug 613: open file handles not closed when creating zarr arrays in cloud storage."""

import fsspec
import numpy as np
import pytest
import zarr


@pytest.mark.skip("Do not need to run issue reproducer regularly")
class TestBug613HandlesNotClosed:
    """Reproducer for bug 613: open file handles not closed when creating zarr arrays in cloud storage.

    zarr: 3.1.2
    Issue Link: https://github.com/TGSAI/mdio-python/issues/613
    """

    gs_file = "gs://tgs-datascience-mdio-dev/drepin/data.tmp"

    def test_zarr_only_file_ok(self) -> None:
        """OK when fsspec is used only via zarr API, no error is observed."""
        # Create a zarr array in the cloud
        arr1 = zarr.create_array(self.gs_file, shape=(2, 2), dtype="int32", overwrite=True)
        arr1[:] = np.array([[1, 2], [3, 4]], dtype="int32")

    def test_zarr_only_store_ok(self) -> None:
        """OK when fsspec is used only via zarr API, no error is observed."""
        # Create a zarr array in the cloud
        # https://zarr.readthedocs.io/en/v3.1.2/api/zarr/storage/index.html#zarr.storage.FsspecStore
        store1: fsspec.AbstractFileSystem = zarr.storage.FsspecStore.from_url(self.gs_file, read_only=False)
        arr1 = zarr.create_array(store1, shape=(2, 2), dtype="int32", overwrite=True)
        arr1[:] = np.array([[1, 2], [3, 4]], dtype="int32")

    def test_zarr_w_fsspec_error(self) -> None:
        """ERROR when Fsspec is used to create AbstractFileSystem, which is passed to Zarr.

        The following error is generated:
            'RuntimeError: Task <Task pending ... > attached to a different loop. Task was destroyed but it is pending!'
        """
        # Create a zarr array in the cloud
        fs, url = fsspec.url_to_fs(self.gs_file)
        store1: fsspec.AbstractFileSystem = fs.get_mapper(url)
        arr1 = zarr.create_array(store1, shape=(2, 2), dtype="int32", overwrite=True)
        arr1[:] = np.array([[1, 2], [3, 4]], dtype="int32")

    def test_fsspec_and_zarr_error(self) -> None:
        """ERROR when an instance of fsspec filesystem is created and zarr fsspec are used in the same thread.

        The following error is generated:
            'RuntimeError: Task <Task pending ... > attached to a different loop. Task was destroyed but it is pending!'
        """
        # 0) Create an instance of fsspec filesystem
        fs: fsspec.AbstractFileSystem = fsspec.filesystem("gs")
        _ = fs.info(self.gs_file)

        # 1) Create a zarr array in the cloud
        store1: fsspec.AbstractFileSystem = zarr.storage.FsspecStore.from_url(self.gs_file, read_only=False)
        arr1 = zarr.create_array(store1, shape=(2, 2), dtype="int32", overwrite=True)
        arr1[:] = np.array([[1, 2], [3, 4]], dtype="int32")
