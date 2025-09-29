"""Test for create_empty_mdio function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from segy.standards import get_segy_standard

if TYPE_CHECKING:
    from pathlib import Path

    from segy.schema import SegySpec
from tests.integration.testing_helpers import get_values
from tests.integration.testing_helpers import validate_variable

from mdio import __version__
from mdio.api.io import open_mdio
from mdio.builder.template_registry import get_template
from mdio.converters.segy import create_empty_mdio
from mdio.core import Dimension
from mdio.core import Grid


class TestCreateEmptyPostStack3DTimeMdio:
    """Tests for create_empty_mdio function."""

    @pytest.fixture(scope="class")
    def segy_spec(self) -> SegySpec:
        """Return the SEG-Y specification for the test."""
        return get_segy_standard(1.0)

    @pytest.fixture(scope="class")
    def empty_mdio_path(self, segy_spec: SegySpec, empty_mdio: Path) -> Path:
        """Create a temporary empty MDIO file for testing.

        This fixture is scoped to the class level, so it will be executed only once
        and shared across all test methods in the class.
        """
        # Create the grid with the specified dimensions
        grid = Grid(
            dims=[
                Dimension(name="inline", coords=range(100, 300, 1)),  # 100-300 with step 1
                Dimension(name="crossline", coords=range(1000, 1600, 2)),  # 1000-1600 with step 2
                Dimension(name="time", coords=range(0, 3000, 4)),  # 0-3 seconds 4ms sample rate
            ]
        )

        mdio_template = get_template("PostStack3DTime")

        # Call create_empty_mdio
        create_empty_mdio(
            segy_spec=segy_spec, 
            mdio_template=mdio_template, 
            grid=grid, 
            output_path=empty_mdio, 
            overwrite=True
        )

        return empty_mdio

    def test_dataset_metadata(self, empty_mdio_path: Path) -> None:
        """Test dataset metadata for empty MDIO file."""
        ds = open_mdio(empty_mdio_path)

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

    def test_grid(self, empty_mdio_path: Path, segy_spec: SegySpec) -> None:
        """Test grid validation for empty MDIO file."""
        ds = open_mdio(empty_mdio_path)

        # Check that the dataset has the expected shape
        assert ds.sizes == {"inline": 200, "crossline": 300, "time": 750}

        # Validate the dimension coordinate variables
        validate_variable(ds, "inline", (200,), ("inline",), np.int32, range(100, 300), get_values)
        validate_variable(ds, "crossline", (300,), ("crossline",), np.int32, range(1000, 1600, 2), get_values)
        validate_variable(ds, "time", (750,), ("time",), np.int32, range(0, 3000, 4), get_values)

        # Validate the non-dimensional coordinate variables (should be empty for empty dataset)
        validate_variable(ds, "cdp_x", (200, 300), ("inline", "crossline"), np.float64, None, None)
        validate_variable(ds, "cdp_y", (200, 300), ("inline", "crossline"), np.float64, None, None)

        # Validate the headers (should be empty for empty dataset)
        # Infer the dtype from segy_spec and ignore endianness
        header_dtype = segy_spec.trace.header.dtype.newbyteorder("native")
        validate_variable(ds, "headers", (200, 300), ("inline", "crossline"), header_dtype, None, None)

        # Validate the trace mask (should be all True for empty dataset)
        validate_variable(ds, "trace_mask", (200, 300), ("inline", "crossline"), np.bool_, None, None)
        trace_mask = ds["trace_mask"].values
        assert np.all(trace_mask), "All traces should be marked as live in empty dataset"

        # Validate the amplitude data (should be empty)
        validate_variable(ds, "amplitude", (200, 300, 750), ("inline", "crossline", "time"), np.float32, None, None)
