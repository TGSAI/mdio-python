"""Tests for generic coordinate population helpers in ingestion."""

from __future__ import annotations

import numpy as np
import pytest
from xarray import DataArray as xr_DataArray
from xarray import Dataset as xr_Dataset

from mdio.ingestion.coordinates import populate_dim_coordinates
from mdio.ingestion.coordinates import populate_non_dim_coordinates
from tests.unit.ingestion.testing_helpers import make_grid
from tests.unit.ingestion.testing_helpers import make_grid_with_map


class TestPopulateDimCoordinates:
    """Tests for ``populate_dim_coordinates``."""

    def test_assigns_coords_for_each_dim(self) -> None:
        """Dim coords should be copied from Grid dims onto the dataset arrays."""
        inline_coords = np.array([10, 20, 30], dtype=np.int32)
        crossline_coords = np.array([100, 200], dtype=np.int32)
        depth_coords = np.array([0, 4, 8, 12], dtype=np.int32)
        grid = make_grid(
            [
                ("inline", inline_coords),
                ("crossline", crossline_coords),
                ("depth", depth_coords),
            ]
        )

        dataset = xr_Dataset(
            {
                "inline": xr_DataArray(np.zeros(3, dtype=np.int32), dims=["inline"]),
                "crossline": xr_DataArray(np.zeros(2, dtype=np.int32), dims=["crossline"]),
                "depth": xr_DataArray(np.zeros(4, dtype=np.int32), dims=["depth"]),
            }
        )

        dataset, drop_vars = populate_dim_coordinates(dataset, grid, drop_vars_delayed=[])

        np.testing.assert_array_equal(dataset["inline"].values, inline_coords)
        np.testing.assert_array_equal(dataset["crossline"].values, crossline_coords)
        np.testing.assert_array_equal(dataset["depth"].values, depth_coords)
        assert drop_vars == ["inline", "crossline", "depth"]

    def test_extends_existing_drop_vars(self) -> None:
        """The drop list should be extended, not replaced."""
        grid = make_grid([("x", np.array([1, 2], dtype=np.int32))])
        dataset = xr_Dataset({"x": xr_DataArray(np.zeros(2, dtype=np.int32), dims=["x"])})

        _, drop_vars = populate_dim_coordinates(dataset, grid, drop_vars_delayed=["already_there"])

        assert drop_vars == ["already_there", "x"]


class TestPopulateNonDimCoordinates:
    """Tests for ``populate_non_dim_coordinates``."""

    def _make_dataset_with_coord(
        self,
        coord_name: str,
        shape: tuple[int, ...],
        dims: tuple[str, ...],
        encoding: dict | None,
        dtype: np.dtype,
    ) -> xr_Dataset:
        data = xr_DataArray(np.zeros(shape, dtype=dtype), dims=list(dims))
        if encoding is not None:
            data.encoding.update(encoding)
        return xr_Dataset({coord_name: data})

    def test_populates_2d_coordinate_with_scaling(self) -> None:
        """Spatial coord ``cdp_x`` should be filled and scaled."""
        inline = np.array([1, 2], dtype=np.int32)
        crossline = np.array([10, 20, 30], dtype=np.int32)
        sample = np.array([0, 4], dtype=np.int32)
        # Inline-major live records → trace indices 0..5 populate the full (2, 3) grid.
        live = [(1, 10), (1, 20), (1, 30), (2, 10), (2, 20), (2, 30)]
        grid = make_grid_with_map(
            [("inline", inline), ("crossline", crossline), ("sample", sample)],
            live_records=live,
        )

        coord_values = np.array([100, 200, 300, 400, 500, 600], dtype=np.float64)
        coordinates = {"cdp_x": coord_values}

        dataset = self._make_dataset_with_coord(
            coord_name="cdp_x",
            shape=(2, 3),
            dims=("inline", "crossline"),
            encoding={"_FillValue": np.float64(-1.0)},
            dtype=np.float64,
        )

        dataset, drop_vars = populate_non_dim_coordinates(
            dataset,
            grid,
            coordinates=coordinates,
            drop_vars_delayed=[],
            spatial_coordinate_scalar=10,
        )

        expected = (coord_values.reshape((2, 3)) * 10).astype(np.float64)
        np.testing.assert_array_equal(dataset["cdp_x"].values, expected)
        assert drop_vars == ["cdp_x"]
        assert coordinates == {}

    def test_uses_fill_value_for_dead_traces(self) -> None:
        """Cells without a live trace should keep the configured fill value."""
        inline = np.array([1, 2], dtype=np.int32)
        crossline = np.array([10, 20], dtype=np.int32)
        sample = np.array([0, 4], dtype=np.int32)
        # Only 3 of 4 cells are live; (inline=1, crossline=20) is dead.
        live = [(1, 10), (2, 10), (2, 20)]
        grid = make_grid_with_map(
            [("inline", inline), ("crossline", crossline), ("sample", sample)],
            live_records=live,
        )

        coord_values = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        dataset = self._make_dataset_with_coord(
            coord_name="cdp_x",
            shape=(2, 2),
            dims=("inline", "crossline"),
            encoding={"_FillValue": np.float64(-9999.0)},
            dtype=np.float64,
        )

        dataset, _ = populate_non_dim_coordinates(
            dataset,
            grid,
            coordinates={"cdp_x": coord_values},
            drop_vars_delayed=[],
            spatial_coordinate_scalar=1,
        )

        expected = np.array([[100.0, -9999.0], [200.0, 300.0]], dtype=np.float64)
        np.testing.assert_array_equal(dataset["cdp_x"].values, expected)

    def test_non_spatial_coordinate_not_scaled(self) -> None:
        """Non-spatial coords (e.g. offset) must not be touched by coord scalar."""
        inline = np.array([1, 2], dtype=np.int32)
        crossline = np.array([10, 20], dtype=np.int32)
        sample = np.array([0, 4], dtype=np.int32)
        live = [(1, 10), (1, 20), (2, 10), (2, 20)]
        grid = make_grid_with_map(
            [("inline", inline), ("crossline", crossline), ("sample", sample)],
            live_records=live,
        )

        coord_values = np.array([5, 6, 7, 8], dtype=np.float64)
        dataset = self._make_dataset_with_coord(
            coord_name="not_spatial",
            shape=(2, 2),
            dims=("inline", "crossline"),
            encoding={"_FillValue": np.float64(0.0)},
            dtype=np.float64,
        )

        dataset, _ = populate_non_dim_coordinates(
            dataset,
            grid,
            coordinates={"not_spatial": coord_values},
            drop_vars_delayed=[],
            spatial_coordinate_scalar=100,  # would change values if applied
        )

        np.testing.assert_array_equal(dataset["not_spatial"].values, coord_values.reshape((2, 2)))

    def test_reduced_coordinate_uses_slice(self) -> None:
        """A coord declared on a subset of dims should be filled via a sliced map."""
        inline = np.array([1, 2], dtype=np.int32)
        crossline = np.array([10, 20, 30], dtype=np.int32)
        sample = np.array([0, 4], dtype=np.int32)
        live = [(1, 10), (1, 20), (1, 30), (2, 10), (2, 20), (2, 30)]
        grid = make_grid_with_map(
            [("inline", inline), ("crossline", crossline), ("sample", sample)],
            live_records=live,
        )

        # Trace indices along the inline=0 row are 0, 1, 2 so the coord values
        # at those positions are taken from coord_values[0:3].
        coord_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64)
        dataset = self._make_dataset_with_coord(
            coord_name="offset",
            shape=(3,),
            dims=("crossline",),
            encoding={"_FillValue": np.float64(-1.0)},
            dtype=np.float64,
        )

        dataset, _ = populate_non_dim_coordinates(
            dataset,
            grid,
            coordinates={"offset": coord_values},
            drop_vars_delayed=[],
            spatial_coordinate_scalar=1,
        )

        np.testing.assert_array_equal(dataset["offset"].values, coord_values[:3])

    def test_default_fill_value_is_nan_when_encoding_missing(self) -> None:
        """When no ``_FillValue`` / ``fill_value`` is set, dead traces become NaN."""
        inline = np.array([1, 2], dtype=np.int32)
        crossline = np.array([10, 20], dtype=np.int32)
        sample = np.array([0, 4], dtype=np.int32)
        live = [(1, 10), (2, 10), (2, 20)]
        grid = make_grid_with_map(
            [("inline", inline), ("crossline", crossline), ("sample", sample)],
            live_records=live,
        )

        coord_values = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        dataset = self._make_dataset_with_coord(
            coord_name="cdp_x",
            shape=(2, 2),
            dims=("inline", "crossline"),
            encoding=None,
            dtype=np.float64,
        )

        dataset, _ = populate_non_dim_coordinates(
            dataset,
            grid,
            coordinates={"cdp_x": coord_values},
            drop_vars_delayed=[],
            spatial_coordinate_scalar=1,
        )

        actual = dataset["cdp_x"].values
        assert np.isnan(actual[0, 1])
        assert actual[0, 0] == pytest.approx(1.5)
        assert actual[1, 0] == pytest.approx(2.5)
        assert actual[1, 1] == pytest.approx(3.5)

    def test_fill_value_key_in_encoding_is_honored(self) -> None:
        """The lowercase ``fill_value`` encoding key must be honored when ``_FillValue`` is absent."""
        inline = np.array([1, 2], dtype=np.int32)
        crossline = np.array([10, 20], dtype=np.int32)
        sample = np.array([0, 4], dtype=np.int32)
        # Dead cell at (inline=1, crossline=20).
        live = [(1, 10), (2, 10), (2, 20)]
        grid = make_grid_with_map(
            [("inline", inline), ("crossline", crossline), ("sample", sample)],
            live_records=live,
        )

        coord_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        dataset = self._make_dataset_with_coord(
            coord_name="cdp_x",
            shape=(2, 2),
            dims=("inline", "crossline"),
            encoding={"fill_value": np.float64(-42.0)},
            dtype=np.float64,
        )

        dataset, _ = populate_non_dim_coordinates(
            dataset,
            grid,
            coordinates={"cdp_x": coord_values},
            drop_vars_delayed=[],
            spatial_coordinate_scalar=1,
        )

        expected = np.array([[1.0, -42.0], [2.0, 3.0]], dtype=np.float64)
        np.testing.assert_array_equal(dataset["cdp_x"].values, expected)

    def test_empty_coordinates_is_noop(self) -> None:
        """An empty coordinates dict should leave the dataset and drop list untouched."""
        inline = np.array([1, 2], dtype=np.int32)
        crossline = np.array([10, 20], dtype=np.int32)
        sample = np.array([0, 4], dtype=np.int32)
        live = [(1, 10), (1, 20), (2, 10), (2, 20)]
        grid = make_grid_with_map(
            [("inline", inline), ("crossline", crossline), ("sample", sample)],
            live_records=live,
        )

        dataset = xr_Dataset()

        dataset, drop_vars = populate_non_dim_coordinates(
            dataset,
            grid,
            coordinates={},
            drop_vars_delayed=["pre_existing"],
            spatial_coordinate_scalar=1,
        )

        assert drop_vars == ["pre_existing"]
        assert len(dataset.data_vars) == 0
