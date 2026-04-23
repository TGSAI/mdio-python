"""Module for testing NumPy to MDIO conversion functionality.

This module contains tests for the `numpy_to_mdio` function, ensuring proper conversion
of NumPy arrays to MDIO format using templates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.testing as npt
import pytest

from mdio.api.io import open_mdio
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
from mdio.converters.numpy import numpy_to_mdio

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MockTemplate(AbstractDatasetTemplate):
    """Mock template for testing."""

    def __init__(self, dim_names: tuple[str, ...], data_domain: str = "time"):
        super().__init__(data_domain=data_domain)
        self._dim_names = dim_names
        self._physical_coord_names = ()
        self._logical_coord_names = ()
        self._var_chunk_shape = (8, 8, 8)

    @property
    def _name(self) -> str:
        return "MockTemplate"

    def _load_dataset_attributes(self) -> dict:
        return {"dataType": "numpy_array", "surveyType": "generic"}


@pytest.fixture
def mock_array() -> NDArray:
    """Make a mock array."""
    rng = np.random.default_rng()
    return rng.uniform(size=(15, 10, 20)).astype("float32")


@pytest.fixture
def mock_template() -> AbstractDatasetTemplate:
    """Make a mock template."""
    return MockTemplate(("inline", "crossline", "time"))


CHUNK_SIZE = (8, 8, 8)


def test_npy_to_mdio_basic(mock_array: NDArray, mock_template: AbstractDatasetTemplate) -> None:
    """Test basic NumPy to MDIO conversion using a template."""
    numpy_to_mdio(mock_array, mock_template, "memory://npy.mdio")
    ds = open_mdio("memory://npy.mdio")

    # Check data
    data_var = ds.attrs.get("defaultVariableName", "amplitude")
    npt.assert_array_equal(ds[data_var].values, mock_array)

    # Check dimensions
    assert list(ds.sizes.keys()) == ["inline", "crossline", "time"]
    assert ds[data_var].shape == mock_array.shape


def test_npy_to_mdio_with_coords(mock_array: NDArray, mock_template: AbstractDatasetTemplate) -> None:
    """Test NumPy to MDIO conversion with custom coordinates."""
    index_coords = {
        "inline": np.arange(100, 115),
        "crossline": np.arange(200, 210),
        "time": np.arange(0, 20),
    }
    numpy_to_mdio(
        mock_array,
        mock_template,
        "memory://npy_coord.mdio",
        index_coords,
    )
    ds = open_mdio("memory://npy_coord.mdio")

    # Check data
    data_var = ds.attrs.get("defaultVariableName", "amplitude")
    npt.assert_array_equal(ds[data_var].values, mock_array)
    assert ds[data_var].shape == mock_array.shape

    # Check coordinates
    npt.assert_array_equal(ds["inline"].values, index_coords["inline"])
    npt.assert_array_equal(ds["crossline"].values, index_coords["crossline"])
    npt.assert_array_equal(ds["time"].values, index_coords["time"])


def test_npy_to_mdio_default_chunksize(mock_array: NDArray, mock_template: AbstractDatasetTemplate) -> None:
    """Test NumPy to MDIO conversion using template's default chunk size."""
    numpy_to_mdio(mock_array, mock_template, "memory://npy_default.mdio")
    ds = open_mdio("memory://npy_default.mdio")

    # Check data
    data_var = ds.attrs.get("defaultVariableName", "amplitude")
    npt.assert_array_equal(ds[data_var].values, mock_array)
    assert list(ds.sizes.keys()) == ["inline", "crossline", "time"]
    assert ds[data_var].shape == mock_array.shape


def test_npy_to_mdio_seismic_template(mock_array: NDArray) -> None:
    """Test NumPy to MDIO conversion using a seismic template."""
    template = Seismic3DPostStackTemplate(data_domain="time")
    numpy_to_mdio(mock_array, template, "memory://npy_seismic.mdio")
    ds = open_mdio("memory://npy_seismic.mdio")

    # Check data
    data_var = ds.attrs.get("defaultVariableName", "amplitude")
    npt.assert_array_equal(ds[data_var].values, mock_array)
    assert list(ds.sizes.keys()) == ["inline", "crossline", "time"]
    assert ds[data_var].shape == mock_array.shape


def test_npy_to_mdio_invalid_coordinate_names(mock_array: NDArray, mock_template: AbstractDatasetTemplate) -> None:
    """Test NumPy to MDIO conversion with invalid coordinate names."""
    index_coords = {
        "inline": np.arange(15),  # valid
        "crossline": np.arange(10),  # valid
        "invalid_dim": np.arange(20),  # invalid dimension name
    }

    with pytest.raises(ValueError, match=r"Coordinate name 'invalid_dim' not found in template dimensions"):
        numpy_to_mdio(mock_array, mock_template, "memory://npy_invalid.mdio", index_coords=index_coords)


def test_npy_to_mdio_invalid_coordinate_sizes(mock_array: NDArray, mock_template: AbstractDatasetTemplate) -> None:
    """Test NumPy to MDIO conversion with invalid coordinate array sizes."""
    index_coords = {
        "inline": np.arange(10),  # wrong size (should be 15)
        "crossline": np.arange(10),  # valid
        "time": np.arange(20),  # valid
    }

    with pytest.raises(
        ValueError, match=r"Size of coordinate 'inline' \(10\) does not match array dimension size \(15\)"
    ):
        numpy_to_mdio(mock_array, mock_template, "memory://npy_wrong_size.mdio", index_coords=index_coords)


def test_npy_to_mdio_valid_custom_coordinates(mock_array: NDArray, mock_template: AbstractDatasetTemplate) -> None:
    """Test NumPy to MDIO conversion with valid custom coordinates."""
    index_coords = {
        "inline": np.arange(100, 115),  # valid size (15)
        "crossline": np.arange(200, 210),  # valid size (10)
        "time": np.arange(0, 20),  # valid size (20)
    }

    numpy_to_mdio(mock_array, mock_template, "memory://npy_valid_custom.mdio", index_coords=index_coords)
    ds = open_mdio("memory://npy_valid_custom.mdio")

    # Check data
    data_var = ds.attrs.get("defaultVariableName", "amplitude")
    npt.assert_array_equal(ds[data_var].values, mock_array)

    # Check coordinates are correctly set
    npt.assert_array_equal(ds["inline"].values, index_coords["inline"])
    npt.assert_array_equal(ds["crossline"].values, index_coords["crossline"])
    npt.assert_array_equal(ds["time"].values, index_coords["time"])
