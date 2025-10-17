"""Test for create_empty_mdio function."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import ScalarType
from segy.standards import get_segy_standard

from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.schemas.v1.units import SpeedUnitEnum
from mdio.builder.schemas.v1.units import SpeedUnitModel
from mdio.builder.schemas.v1.units import TimeUnitEnum
from mdio.builder.schemas.v1.units import TimeUnitModel

if TYPE_CHECKING:
    from pathlib import Path

    from xarray import Dataset as xr_Dataset


from tests.integration.test_segy_roundtrip_teapot import get_teapot_segy_spec
from tests.integration.testing_helpers import get_values
from tests.integration.testing_helpers import validate_variable

from mdio import __version__
from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.builder.schemas.v1.stats import CenteredBinHistogram
from mdio.builder.schemas.v1.stats import SummaryStatistics
from mdio.converters.mdio import mdio_to_segy
from mdio.core import Dimension
from mdio.creators.mdio import create_empty_like


@pytest.mark.order(1000)
class TestCreateEmptyPostStack3DTimeMdio:
    """Tests for create_empty_mdio function."""

    @classmethod
    def _get_customized_v10_trace_header_spec(cls) -> HeaderSpec:
        """Get the header spec for the MDIO dataset."""
        trace_header_fields = [
            HeaderField(name="inline", byte=17, format=ScalarType.INT32),
            HeaderField(name="crossline", byte=13, format=ScalarType.INT32),
            HeaderField(name="cdp_x", byte=181, format=ScalarType.INT32),
            HeaderField(name="cdp_y", byte=185, format=ScalarType.INT32),
            HeaderField(name="coordinate_scalar", byte=71, format=ScalarType.INT16),
        ]
        hs: HeaderSpec = get_segy_standard(1.0).trace.header
        hs.customize(fields=trace_header_fields)
        return hs

    @classmethod
    def _validate_dataset_metadata(cls, ds: xr_Dataset) -> None:
        """Validate the dataset metadata."""
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
        assert "gridOverrides" not in attributes, "Empty dataset should not have gridOverrides"

    @classmethod
    def _validate_empty_mdio_dataset(cls, ds: xr_Dataset, has_headers: bool) -> None:
        """Validate an empty MDIO dataset structure and content."""
        # Check that the dataset has the expected shape
        assert ds.sizes == {"inline": 345, "crossline": 188, "time": 1501}

        # Validate the dimension coordinate variables
        validate_variable(ds, "inline", (345,), ("inline",), np.int32, range(1, 346), get_values)
        validate_variable(ds, "crossline", (188,), ("crossline",), np.int32, range(1, 189), get_values)
        validate_variable(ds, "time", (1501,), ("time",), np.int32, range(0, 3002, 2), get_values)

        # Validate the non-dimensional coordinate variables (should be empty for empty dataset)
        validate_variable(ds, "cdp_x", (345, 188), ("inline", "crossline"), np.float64, None, None)
        validate_variable(ds, "cdp_y", (345, 188), ("inline", "crossline"), np.float64, None, None)

        if has_headers:
            segy_spec = get_teapot_segy_spec()
            # Validate the headers (should be empty for empty dataset)
            # Infer the dtype from segy_spec and ignore endianness
            header_dtype = segy_spec.trace.header.dtype.newbyteorder("native")
            validate_variable(ds, "headers", (345, 188), ("inline", "crossline"), header_dtype, None, None)
            validate_variable(ds, "segy_file_header", (), (), np.dtype("U1"), None, None)

            assert "segy_file_header" in ds.variables
            assert ds["segy_file_header"].attrs.get("textHeader", None) is None, (
                "TextHeader should be empty for empty dataset"
            )
            assert ds["segy_file_header"].attrs.get("binaryHeader", None) is None, (
                "BinaryHeader should be empty for empty dataset"
            )
            assert ds["segy_file_header"].attrs.get("rawBinaryHeader", None) is None, (
                "RawBinaryHeader should be empty for empty dataset"
            )
        else:
            assert "headers" not in ds.variables
            assert "segy_file_header" not in ds.variables

        # Validate the trace mask
        validate_variable(ds, "trace_mask", (345, 188), ("inline", "crossline"), np.bool_, None, None)
        trace_mask = ds["trace_mask"].values
        assert not np.any(trace_mask), "All traces should be marked as dead in empty dataset"

        # Validate the amplitude data (should be empty)
        validate_variable(ds, "amplitude", (345, 188, 1501), ("inline", "crossline", "time"), np.float32, None, None)
        assert ds["amplitude"].attrs.get("statsV1", None) is None, "StatsV1 should be empty for empty dataset"
        assert ds["amplitude"].attrs.get("unitsV1", None) is None, "UnitsV1 should be empty for empty dataset"

    @pytest.mark.order(1001)
    @pytest.mark.dependency
    def test_create_empty_like(self, teapot_mdio_tmp: Path, empty_mdio_with_headers: Path) -> None:
        """Create an empty MDIO file like the input file."""
        _ = empty_mdio_with_headers
        ds = create_empty_like(
            input_path=teapot_mdio_tmp,
            output_path=None,  # We don't want to write to disk for now
            keep_coordinates=True,
            overwrite=True,
        )
        self._validate_dataset_metadata(ds)
        self._validate_empty_mdio_dataset(ds, has_headers=True)
