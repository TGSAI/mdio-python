"""End to end testing for OBN SEG-Y to MDIO conversion with grid overrides."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import dask
import pytest
import xarray.testing as xrt
from tests.integration.conftest import get_segy_mock_obn_spec

from mdio.api.io import open_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.converters.segy import segy_to_mdio

if TYPE_CHECKING:
    from pathlib import Path

dask.config.set(scheduler="synchronous")
os.environ["MDIO__IMPORT__SAVE_SEGY_FILE_HEADER"] = "true"


class TestImportObnWithComponent:
    """Test OBN SEG-Y import with component header (standard case)."""

    def test_import_obn_with_calculate_shot_index(
        self,
        segy_mock_obn_with_component: Path,
        zarr_tmp: Path,
    ) -> None:
        """Test importing OBN SEG-Y with CalculateShotIndex grid override."""
        segy_spec = get_segy_mock_obn_spec(include_component=True)
        grid_override = {"CalculateShotIndex": True}

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get("ObnReceiverGathers3D"),
            input_path=segy_mock_obn_with_component,
            output_path=zarr_tmp,
            overwrite=True,
            grid_overrides=grid_override,
        )

        # Expected values
        num_samples = 25
        components = [1, 2, 3, 4]
        receivers = [101, 102, 103]
        shot_lines = [1, 2]
        guns = [1, 2]

        ds = open_mdio(zarr_tmp)

        assert ds["segy_file_header"].attrs["binaryHeader"]["samples_per_trace"] == num_samples
        assert ds.attrs["attributes"]["gridOverrides"] == grid_override

        # Check dimension coordinates
        xrt.assert_duckarray_equal(ds["component"], components)
        xrt.assert_duckarray_equal(ds["receiver"], receivers)
        xrt.assert_duckarray_equal(ds["shot_line"], shot_lines)
        xrt.assert_duckarray_equal(ds["gun"], guns)

        # shot_index should be calculated (0-based indices)
        # With interleaved geometry: gun1: 1,3,5 -> indices 0,1,2; gun2: 2,4,6 -> indices 1,2,3
        # Combined unique indices: 0, 1, 2, 3
        expected_shot_index = [0, 1, 2, 3]
        xrt.assert_duckarray_equal(ds["shot_index"], expected_shot_index)

        # Check time coordinate
        times_expected = list(range(0, num_samples, 1))
        xrt.assert_duckarray_equal(ds["time"], times_expected)

        # Check that shot_point is preserved as a coordinate (not a dimension)
        assert "shot_point" in ds.coords
        assert ds["shot_point"].dims == ("shot_line", "gun", "shot_index")


class TestImportObnSyntheticComponent:
    """Test OBN SEG-Y import without component header - component is synthesized."""

    def test_import_obn_synthetic_component(
        self,
        segy_mock_obn_no_component: Path,
        zarr_tmp: Path,
    ) -> None:
        """Test importing OBN SEG-Y without component - component is automatically synthesized."""
        segy_spec = get_segy_mock_obn_spec(include_component=False)
        grid_override = {"CalculateShotIndex": True}

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get("ObnReceiverGathers3D"),
            input_path=segy_mock_obn_no_component,
            output_path=zarr_tmp,
            overwrite=True,
            grid_overrides=grid_override,
        )

        # Expected values
        num_samples = 25
        receivers = [101, 102, 103]
        shot_lines = [1, 2]
        guns = [1, 2]

        ds = open_mdio(zarr_tmp)

        assert ds["segy_file_header"].attrs["binaryHeader"]["samples_per_trace"] == num_samples
        assert ds.attrs["attributes"]["gridOverrides"] == grid_override

        # Component should be a dimension with synthesized value [1]
        assert "component" in ds.dims
        xrt.assert_duckarray_equal(ds["component"], [1])  # Synthesized with default value 1

        # Check other dimension coordinates
        xrt.assert_duckarray_equal(ds["receiver"], receivers)
        xrt.assert_duckarray_equal(ds["shot_line"], shot_lines)
        xrt.assert_duckarray_equal(ds["gun"], guns)

        # shot_index should be calculated
        expected_shot_index = [0, 1, 2, 3]
        xrt.assert_duckarray_equal(ds["shot_index"], expected_shot_index)

        # Check time coordinate
        times_expected = list(range(0, num_samples, 1))
        xrt.assert_duckarray_equal(ds["time"], times_expected)

        # Check that shot_point is preserved as a coordinate
        assert "shot_point" in ds.coords
        assert ds["shot_point"].dims == ("shot_line", "gun", "shot_index")


class TestImportObnMissingCalculateShotIndex:
    """Test OBN SEG-Y import without CalculateShotIndex grid override."""

    def test_import_obn_without_calculate_shot_index_raises(
        self,
        segy_mock_obn_with_component: Path,
        zarr_tmp: Path,
    ) -> None:
        """Test that importing OBN SEG-Y without CalculateShotIndex raises ValueError.

        The OBN template has shot_index as a calculated dimension. Without the
        CalculateShotIndex grid override, the shot_index field is not computed, and
        the import should fail with a clear error message.
        """
        segy_spec = get_segy_mock_obn_spec(include_component=True)

        with pytest.raises(ValueError, match=r"Required computed fields.*not found after grid overrides") as exc_info:
            segy_to_mdio(
                segy_spec=segy_spec,
                mdio_template=TemplateRegistry().get("ObnReceiverGathers3D"),
                input_path=segy_mock_obn_with_component,
                output_path=zarr_tmp,
                overwrite=True,
                grid_overrides=None,  # No CalculateShotIndex
            )

        error_message = str(exc_info.value)
        assert "shot_index" in error_message
        assert "ObnReceiverGathers3D" in error_message
