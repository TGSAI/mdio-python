"""End to end testing for SEG-Y to MDIO conversion and back."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import dask
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from segy import SegyFile


from mdio import mdio_to_segy
from mdio.converters import segy_to_mdio_v1

from mdio.converters.segy_to_mdio_v1 import StorageLocation
from mdio.core import Dimension
from mdio.segy.compat import mdio_segy_spec


if TYPE_CHECKING:
    from pathlib import Path

dask.config.set(scheduler="synchronous")

@pytest.mark.dependency
@pytest.mark.parametrize("index_bytes", [(17, 13)])
@pytest.mark.parametrize("index_names", [("inline", "crossline")])
def test_3d_import_v1(
    segy_input: Path,
    zarr_tmp: Path,
    index_bytes: tuple[int, ...],
    index_names: tuple[str, ...],
) -> None:
    """Test importing a SEG-Y file to MDIO."""

    segy_to_mdio_v1(
        segy_spec= float("1.0"),
        mdio_template= "PostStack3DTime",
        index_bytes=index_bytes,
        index_names=index_names,
        index_types=None,
        input= StorageLocation(segy_input.__str__()),
        output= StorageLocation(zarr_tmp.__str__()),
        overwrite=True
    )

    # Load Xarray dataset from the MDIO file

    pass

@pytest.mark.dependency("test_3d_import_v1")
class TestReaderV1:
    """Test reader functionality."""

    def test_meta_read(self, zarr_tmp: Path) -> None:
        """Metadata reading tests."""
        dataset = xr.open_dataset(zarr_tmp.__str__(), engine="zarr")

        # assert mdio.binary_header["samples_per_trace"] == 1501
        # assert mdio.binary_header["sample_interval"] == 2000

    # def test_grid(self, zarr_tmp: Path) -> None:
    #     """Grid reading tests."""
    #     mdio = MDIOReader(zarr_tmp.__str__())
    #     grid = mdio.grid

    #     assert grid.select_dim("inline") == Dimension(range(1, 346), "inline")
    #     assert grid.select_dim("crossline") == Dimension(range(1, 189), "crossline")
    #     assert grid.select_dim("sample") == Dimension(range(0, 3002, 2), "sample")

    # def test_get_data(self, zarr_tmp: Path) -> None:
    #     """Data retrieval tests."""
    #     mdio = MDIOReader(zarr_tmp.__str__())

    #     assert mdio.shape == (345, 188, 1501)
    #     assert mdio[0, :, :].shape == (188, 1501)
    #     assert mdio[:, 0, :].shape == (345, 1501)
    #     assert mdio[:, :, 0].shape == (345, 188)

    # def test_inline(self, zarr_tmp: Path) -> None:
    #     """Read and compare every 75 inlines' mean and std. dev."""
    #     mdio = MDIOReader(zarr_tmp.__str__())

    #     inlines = mdio[::75, :, :]
    #     mean, std = inlines.mean(), inlines.std()

    #     npt.assert_allclose([mean, std], [1.0555277e-04, 6.0027051e-01])

    # def test_crossline(self, zarr_tmp: Path) -> None:
    #     """Read and compare every 75 crosslines' mean and std. dev."""
    #     mdio = MDIOReader(zarr_tmp.__str__())

    #     xlines = mdio[:, ::75, :]
    #     mean, std = xlines.mean(), xlines.std()

    #     npt.assert_allclose([mean, std], [-5.0329847e-05, 5.9406823e-01])

    # def test_zslice(self, zarr_tmp: Path) -> None:
    #     """Read and compare every 225 z-slices' mean and std. dev."""
    #     mdio = MDIOReader(zarr_tmp.__str__())

    #     slices = mdio[:, :, ::225]
    #     mean, std = slices.mean(), slices.std()

    #     npt.assert_allclose([mean, std], [0.005236923, 0.61279935])
