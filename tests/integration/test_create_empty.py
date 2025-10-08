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


from tests.integration.testing_helpers import get_values
from tests.integration.testing_helpers import validate_variable

from mdio import __version__
from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.builder.schemas.v1.stats import CenteredBinHistogram
from mdio.builder.schemas.v1.stats import SummaryStatistics
from mdio.converters.mdio import mdio_to_segy
from mdio.core import Dimension
from mdio.creators.mdio import create_empty


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
            # Validate the headers (should be empty for empty dataset)
            # Infer the dtype from segy_spec and ignore endianness
            header_dtype = cls._get_customized_v10_trace_header_spec().dtype.newbyteorder("native")
            validate_variable(ds, "headers", (345, 188), ("inline", "crossline"), header_dtype, None, None)
            validate_variable(ds, "segy_file_header", (), (), np.dtype("U1"), None, None)
        else:
            assert "headers" not in ds.variables
            assert "segy_file_header" not in ds.variables

        # Validate the trace mask (should be all True for empty dataset)
        validate_variable(ds, "trace_mask", (345, 188), ("inline", "crossline"), np.bool_, None, None)
        trace_mask = ds["trace_mask"].values
        assert not np.any(trace_mask), "All traces should be marked as dead in empty dataset"

        # Validate the amplitude data (should be empty)
        validate_variable(ds, "amplitude", (345, 188, 1501), ("inline", "crossline", "time"), np.float32, None, None)

    @classmethod
    def _create_empty_mdio(cls, create_headers: bool, output_path: Path, overwrite: bool = True) -> None:
        """Create a temporary empty MDIO file for testing."""
        # Create the grid with the specified dimensions
        dims = [
            Dimension(name="inline", coords=range(1, 346, 1)),  # 100-300 with step 1
            Dimension(name="crossline", coords=range(1, 189, 1)),  # 1000-1600 with step 2
            Dimension(name="time", coords=range(0, 3002, 2)),  # 0-3 seconds 4ms sample rate
        ]

        # If later on, we want to export to SEG-Y, we need to provide the trace header spec.
        # The HeaderSpec can be either standard or customized.
        headers = cls._get_customized_v10_trace_header_spec() if create_headers else None
        create_empty(
            mdio_template_name="PostStack3DTime",
            dimensions=dims,
            output_path=output_path,
            headers=headers,
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
        garbage_dir.mkdir(exist_ok=True)
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

    def test_populate_empty_dataset(self, mdio_with_headers: Path) -> None:
        """Test showing how to populate empty dataset."""
        # Open an empty PostStack3DTime dataset with SEG-Y 1.0 headers
        # NOTES:
        # When this empty dataset was created from the 'PostStack3DTime' template and dimensions,
        # * 'inline', 'crossline', and 'time' dimension coordinate variables were created and pre-populated
        # * 'cdp_x', 'cdp_y' non-dimensional coordinate variables were created
        # * 'amplitude' variable was created (the name of this variable is specified in the template)
        #   HACK: in this example, we will use this variable to store the velocity data
        # * 'trace_mask' variable was created and pre-populated with 'False' fill values
        #   (all traces are marked as dead)
        # * 'headers' and 'segy_file_header' variables were created (if the dataset was created with
        #   headers not None). The 'headers' variable structured datatype is defined by the HeaderSpec
        # that was used to create the empty MDIO
        # * dataset attribute called 'attributes' was created
        ds = open_mdio(mdio_with_headers)

        # 1) Populate dataset's velocity
        var_name = ds.attrs["attributes"]["defaultVariableName"]
        velocity = ds[var_name]
        velocity[:5, :, :] = 1
        velocity[5:10, :, :] = 2
        velocity[50:100, :, :] = 3
        velocity[150:175, :, :] = -1

        # 2) Populate dataset's velocity statistics (optional)
        nonzero_samples = np.ma.masked_invalid(velocity, copy=False)
        stats = SummaryStatistics(
            count=nonzero_samples.count(),
            min=nonzero_samples.min(),
            max=nonzero_samples.max(),
            sum=nonzero_samples.sum(dtype="float64"),
            sum_squares=(np.ma.power(nonzero_samples, 2).sum(dtype="float64")),
            histogram=CenteredBinHistogram(bin_centers=[], counts=[]),
        )
        velocity.attrs["statsV1"] = stats.model_dump(mode="json")

        # 3) Populate the non-dimensional coordinate variables 'cdp_x' and 'cdp_y' (optional)
        origin = [270000, 3290000]  # survey x, y origin
        inline_azimuth_rad = 0.523599  # survey orientation, in radians, from the north to the east (30 degrees)
        spacing = [50, 50]  # survey inline, crossline spacing
        inline_grid, xline_grid = np.meshgrid(ds.inline.values, ds.crossline.values, indexing="ij")
        sin_azimuth = math.sin(inline_azimuth_rad)
        cos_azimuth = math.cos(inline_azimuth_rad)
        ds.cdp_x[:] = origin[0] + inline_grid * spacing[0] * sin_azimuth + xline_grid * spacing[1] * cos_azimuth
        ds.cdp_y[:] = origin[1] + inline_grid * spacing[0] * cos_azimuth - xline_grid * spacing[1] * sin_azimuth

        # 4) Populate dataset's trace mask (optional)
        ds.trace_mask[:] = ~np.isnan(velocity[:, :, 0])

        # 5) Set coordinate and data variable units (optional)
        ds.time.attrs["unitsV1"] = TimeUnitModel(time=TimeUnitEnum.MILLISECOND).model_dump(mode="json")

        ds.cdp_x.attrs["unitsV1"] = LengthUnitModel(length=LengthUnitEnum.FOOT).model_dump(mode="json")
        ds.cdp_x.attrs["unitsV1"] = LengthUnitModel(length=LengthUnitEnum.FOOT).model_dump(mode="json")

        velocity.attrs["unitsV1"] = SpeedUnitModel(speed=SpeedUnitEnum.FEET_PER_SECOND).model_dump(mode="json")

        # 6) Populate dataset's segy trace headers, if those were created (required only if we want to export to SEG-Y)
        if "headers" in ds.variables:
            # Both the structured "headers" and the dummy "segy_file_header" variables are
            # required to enable SEG-Y to MDIO conversion

            # Populate the structured trace "headers" variable
            ds["headers"].values["inline"] = inline_grid
            ds["headers"].values["crossline"] = xline_grid
            # coordinate_scalar:
            # Scalar to be applied to all coordinates specified in Standard Trace Header bytes
            # 73–88 and to bytes Trace Header 181–188 to give the real value. Scalar = 1,
            # ±10, ±100, ±1000, or ±10,000. If positive, scalar is used as a multiplier; if
            # negative, scalar is used as divisor. A value of zero is assumed to be a scalar
            # value of 1.
            ds["headers"].values["coordinate_scalar"][:] = np.int16(-100)
            ds["headers"].values["cdp_x"][:] = np.int32(ds.cdp_x * 100)
            ds["headers"].values["cdp_y"][:] = np.int32(ds.cdp_y * 100)

            # Fill its metadata (.attrs) with 'textHeader' and 'binaryHeader'.
            ds["segy_file_header"].attrs.update(
                {
                    "textHeader": "\n".join(
                        [
                            "C01 BYTES 13-16: CROSSLINE         " + " " * 47,
                            "C02 BYTES 17-20: INLINE            " + " " * 47,
                            "C03 BYTES 71-74: COORDINATE SCALAR " + " " * 47,
                            "C04 BYTES 181-184: CDP X           " + " " * 47,
                            "C05 BYTES 185-188: CDP Y           " + " " * 47,
                            *(f"C{i:02d}" + " " * 77 for i in range(6, 41)),
                        ]
                    ),
                    "binaryHeader": {
                        "data_sample_format": 1,
                        "sample_interval": int(ds.time[1] - ds.time[0]),
                        "samples_per_trace": ds.time.size,
                        "segy_revision_major": 0,
                        "segy_revision_minor": 0,
                    },
                }
            )

        # 7) Create dataset's custom attributes (optional)
        ds.attrs["attributes"]["createdBy"] = "John Doe"

        # 8) Export to MDIO
        output_path_mdio = mdio_with_headers.parent / "populated_empty.mdio"
        to_mdio(ds, output_path=output_path_mdio, mode="w", compute=True)

        # 9) Convert the populated emptyMDIO to SEG-Y
        if "headers" in ds.variables:
            # Select the SEG-Y standard to use for the conversion
            custom_segy_spec = get_segy_standard(1.0)
            # Customize to use the same HeaderSpec that was used to create the empty MDIO
            custom_segy_spec.trace.header = self._get_customized_v10_trace_header_spec()
            # Convert the MDIO file to SEG-Y
            mdio_to_segy(
                segy_spec=custom_segy_spec,
                input_path=output_path_mdio,
                output_path=mdio_with_headers.parent / "populated_empty.sgy",
            )
