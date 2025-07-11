# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 dataset_serializer public API."""

import xarray as xr

from mdio.schemas.dtype import ScalarType
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.dataset_serializer import to_xarray_dataset
from mdio.schemas.v1.dataset_serializer import to_zarr

from .helpers import make_campos_3d_acceptance_dataset


def test_to_xarray_dataset(capsys) -> None:
    """Test building a complete dataset."""
    dataset = (
        MDIODatasetBuilder("test_dataset")
        .add_dimension("inline", 100)
        .add_dimension("crossline", 200)
        .add_dimension("depth", 300)
        .add_coordinate("inline", dimensions=["inline"], data_type=ScalarType.FLOAT64)
        .add_coordinate("crossline", dimensions=["crossline"], data_type=ScalarType.FLOAT64)
        .add_coordinate("x_coord", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT32)
        .add_coordinate("y_coord", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT32)
        .add_variable(
            "data",
            long_name="Test Data",
            dimensions=["inline", "crossline", "depth"],
            coordinates=["inline", "crossline", "x_coord", "y_coord"],
            data_type=ScalarType.FLOAT32,
        )
        .build()
    )

    # with capsys.disabled():
    xds: xr.Dataset = to_xarray_dataset(dataset)

    file_name = "sample_dataset"
    to_zarr(xds, f"test-data/{file_name}.zarr", mode="w")


def test_campos_3d_acceptance_to_xarray_dataset(capsys) -> None:
    """Test building a complete dataset."""
    dataset = make_campos_3d_acceptance_dataset()

    xds: xr.Dataset = to_xarray_dataset(dataset)

    # file_name = "XYZ"
    file_name = f"{xds.attrs['name']}"
    to_zarr(xds, f"test-data/{file_name}.zarr", mode="w")
