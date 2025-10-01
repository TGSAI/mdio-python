"""Test for create_empty_mdio function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from segy.standards import get_segy_standard

if TYPE_CHECKING:
    from pathlib import Path

    from xarray import Dataset as xr_Dataset

from tests.integration.testing_helpers import get_values
from tests.integration.testing_helpers import validate_variable

from mdio import __version__
from mdio.api.io import open_mdio
from mdio.core import Dimension
from mdio.creators.mdio import create_empty_mdio


class TestCreateEmptyPostStack3DTimeMdio:
    """Tests for create_empty_mdio function."""

    @classmethod
    def _validate_empty_mdio_dataset(cls, ds: xr_Dataset, has_headers: bool) -> None:
        """Validate an empty MDIO dataset structure and content."""
        # Check that the dataset has the expected shape
        assert ds.sizes == {"inline": 200, "crossline": 300, "time": 750}

        # Validate the dimension coordinate variables
        validate_variable(ds, "inline", (200,), ("inline",), np.int32, range(100, 300), get_values)
        validate_variable(ds, "crossline", (300,), ("crossline",), np.int32, range(1000, 1600, 2), get_values)
        validate_variable(ds, "time", (750,), ("time",), np.int32, range(0, 3000, 4), get_values)

        # Validate the non-dimensional coordinate variables (should be empty for empty dataset)
        validate_variable(ds, "cdp_x", (200, 300), ("inline", "crossline"), np.float64, None, None)
        validate_variable(ds, "cdp_y", (200, 300), ("inline", "crossline"), np.float64, None, None)

        if has_headers:
            # Validate the headers (should be empty for empty dataset)
            # Infer the dtype from segy_spec and ignore endianness
            header_dtype = get_segy_standard(1.0).trace.header.dtype.newbyteorder("native")
            validate_variable(ds, "headers", (200, 300), ("inline", "crossline"), header_dtype, None, None)
        else:
            assert "headers" not in ds.variables

        # Validate the trace mask (should be all True for empty dataset)
        validate_variable(ds, "trace_mask", (200, 300), ("inline", "crossline"), np.bool_, None, None)
        trace_mask = ds["trace_mask"].values
        assert np.all(trace_mask), "All traces should be marked as live in empty dataset"

        # Validate the amplitude data (should be empty)
        validate_variable(ds, "amplitude", (200, 300, 750), ("inline", "crossline", "time"), np.float32, None, None)

    @classmethod
    def _create_empty_mdio(cls, create_headers: bool, output_path: Path, overwrite: bool = True) -> None:
        """Create a temporary empty MDIO file for testing."""
        # Create the grid with the specified dimensions
        dims = [
            Dimension(name="inline", coords=range(100, 300, 1)),  # 100-300 with step 1
            Dimension(name="crossline", coords=range(1000, 1600, 2)),  # 1000-1600 with step 2
            Dimension(name="time", coords=range(0, 3000, 4)),  # 0-3 seconds 4ms sample rate
        ]

        # Call create_empty_mdio
        create_empty_mdio(
            mdio_template_name="PostStack3DTime",
            dimensions=dims,
            output_path=output_path,
            create_headers=create_headers,
            overwrite=overwrite,
        )

    @pytest.fixture(scope="class")
    def mdio_with_headers(self, empty_mdio_dir: Path) -> Path:
        """Create a temporary empty MDIO file for testing.

        This fixture is scoped to the class level, so it will be executed only once
        and shared across all test methods in the class.
        """
        empty_mdio: Path = empty_mdio_dir / "with_headers.mdio"
        self._create_empty_mdio(create_headers=True, output_path=empty_mdio)
        return empty_mdio

    @pytest.fixture(scope="class")
    def mdio_no_headers(self, empty_mdio_dir: Path) -> Path:
        """Create a temporary empty MDIO file for testing.

        This fixture is scoped to the class level, so it will be executed only once
        and shared across all test methods in the class.
        """
        empty_mdio: Path = empty_mdio_dir / "no_headers.mdio"
        self._create_empty_mdio(create_headers=False, output_path=empty_mdio)
        return empty_mdio

    def test_dataset_metadata(self, mdio_with_headers: Path) -> None:
        """Test dataset metadata for empty MDIO file."""
        ds = open_mdio(mdio_with_headers)

        # Check basic metadata attributes
        expected_attrs = {
            "apiVersion": __version__,
            "name": "PostStack3DTime",
        }
        actual_attrs_json = ds.attrs

        # Compare one by one due to ever changing createdOn
        for key, value in expected_attrs.items():
            assert key in actual_attrs_json
            if key == "createdOn":
                assert actual_attrs_json[key] is not None
            else:
                assert actual_attrs_json[key] == value

        # Check that createdOn exists
        assert "createdOn" in actual_attrs_json
        assert actual_attrs_json["createdOn"] is not None

        # Validate template attributes
        attributes = ds.attrs["attributes"]
        assert attributes is not None
        assert len(attributes) == 3
        # Validate all attributes provided by the abstract template
        assert attributes["defaultVariableName"] == "amplitude"
        assert attributes["surveyType"] == "3D"
        assert attributes["gatherType"] == "stacked"

    def test_variables(self, mdio_with_headers: Path, mdio_no_headers: Path) -> None:
        """Test grid validation for empty MDIO file."""
        ds = open_mdio(mdio_with_headers)
        self._validate_empty_mdio_dataset(ds, has_headers=True)

        ds = open_mdio(mdio_no_headers)
        self._validate_empty_mdio_dataset(ds, has_headers=False)

    def test_overwrite_behavior(self, empty_mdio_dir: Path) -> None:
        """Test overwrite parameter behavior in create_empty_mdio."""
        empty_mdio = empty_mdio_dir / "empty.mdio"
        empty_mdio.mkdir(parents=True, exist_ok=True)
        garbage_file = empty_mdio / "garbage.txt"
        garbage_file.write_text("This is garbage data that should be overwritten")
        garbage_dir = empty_mdio / "garbage_dir"
        garbage_dir.mkdir()
        (garbage_dir / "nested_garbage.txt").write_text("More garbage")

        # Verify the directory exists with garbage data
        assert empty_mdio.exists()
        assert garbage_file.exists()
        assert garbage_dir.exists()

        # Second call: Try to create MDIO with overwrite=False - should raise FileExistsError
        with pytest.raises(FileExistsError, match="Output location.*exists"):
            self._create_empty_mdio(create_headers=True, output_path=empty_mdio, overwrite=False)

        # Third call: Create MDIO with overwrite=True - should succeed and overwrite garbage
        self._create_empty_mdio(create_headers=True, output_path=empty_mdio, overwrite=True)

        # Validate that the MDIO file can be loaded correctly using the helper function
        ds = open_mdio(empty_mdio)
        self._validate_empty_mdio_dataset(ds, has_headers=True)

        # Verify the garbage data was overwritten (should not exist)
        assert not garbage_file.exists(), "Garbage file should have been overwritten"
        assert not garbage_dir.exists(), "Garbage directory should have been overwritten"
