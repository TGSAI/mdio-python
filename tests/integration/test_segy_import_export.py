"""End to end testing for SEG-Y to MDIO conversion and back."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import dask
import numpy as np
import numpy.testing as npt
import pytest
import xarray.testing as xrt
from segy import SegyFile
from tests.integration.conftest import get_segy_mock_4d_spec
from tests.integration.testing_data import binary_header_teapot_dome
from tests.integration.testing_data import custom_teapot_dome_segy_spec
from tests.integration.testing_data import text_header_teapot_dome
from tests.integration.testing_helpers import get_inline_header_values
from tests.integration.testing_helpers import get_values
from tests.integration.testing_helpers import validate_variable

from mdio import __version__
from mdio import mdio_to_segy
from mdio.api.io import open_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.converters.exceptions import GridTraceSparsityError
from mdio.converters.segy import segy_to_mdio
from mdio.segy.geometry import StreamerShotGeometryType

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from segy.schema import SegySpec

dask.config.set(scheduler="synchronous")


# TODO(Altay): Finish implementing these grid overrides.
# https://github.com/TGSAI/mdio-python/issues/612
@pytest.mark.skip(reason="NonBinned and HasDuplicates haven't been properly implemented yet.")
@pytest.mark.parametrize("grid_override", [{"NonBinned": True}, {"HasDuplicates": True}])
@pytest.mark.parametrize("chan_header_type", [StreamerShotGeometryType.C])
class TestImport4DNonReg:
    """Test for 4D segy import with grid overrides."""

    def test_import_4d_segy(  # noqa: PLR0913
        self,
        segy_mock_4d_shots: dict[StreamerShotGeometryType, Path],
        zarr_tmp: Path,
        grid_override: dict[str, Any],
        chan_header_type: StreamerShotGeometryType,
    ) -> None:
        """Test importing a SEG-Y file to MDIO."""
        segy_spec: SegySpec = get_segy_mock_4d_spec()
        segy_path = segy_mock_4d_shots[chan_header_type]

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get("PreStackShotGathers3DTime"),
            input_path=segy_path,
            output_path=zarr_tmp,
            overwrite=True,
            grid_overrides=grid_override,
        )

        # Expected values
        num_samples = 25
        shots = [2, 3, 5, 6, 7, 8, 9]
        cables = [0, 101, 201, 301]
        receivers_per_cable = [1, 5, 7, 5]

        ds = open_mdio(zarr_tmp)

        assert ds["segy_file_header"].attrs["binaryHeader"]["samples_per_trace"] == num_samples
        assert ds.attrs["attributes"]["gridOverrides"] == grid_override

        assert npt.assert_array_equal(ds["shot_point"], shots)
        xrt.assert_duckarray_equal(ds["cable"], cables)

        # assert grid.select_dim("trace") == Dimension(range(1, np.amax(receivers_per_cable) + 1), "trace")
        expected = list(range(1, np.amax(receivers_per_cable) + 1))
        xrt.assert_duckarray_equal(ds["trace"], expected)

        times_expected = list(range(0, num_samples, 1))
        xrt.assert_duckarray_equal(ds["time"], times_expected)


@pytest.mark.parametrize("grid_override", [{"AutoChannelWrap": True}, None])
@pytest.mark.parametrize("chan_header_type", [StreamerShotGeometryType.A, StreamerShotGeometryType.B])
class TestImport4D:
    """Test for 4D segy import with grid overrides."""

    def test_import_4d_segy(  # noqa: PLR0913
        self,
        segy_mock_4d_shots: dict[StreamerShotGeometryType, Path],
        zarr_tmp: Path,
        grid_override: dict[str, Any],
        chan_header_type: StreamerShotGeometryType,
    ) -> None:
        """Test importing a SEG-Y file to MDIO."""
        segy_spec: SegySpec = get_segy_mock_4d_spec()
        segy_path = segy_mock_4d_shots[chan_header_type]

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get("PreStackShotGathers3DTime"),
            input_path=segy_path,
            output_path=zarr_tmp,
            overwrite=True,
            grid_overrides=grid_override,
        )

        # Expected values
        num_samples = 25
        shots = [2, 3, 5, 6, 7, 8, 9]
        cables = [0, 101, 201, 301]
        receivers_per_cable = [1, 5, 7, 5]

        ds = open_mdio(zarr_tmp)

        assert ds["segy_file_header"].attrs["binaryHeader"]["samples_per_trace"] == num_samples
        assert ds.attrs["attributes"].get("gridOverrides", None) == grid_override  # may not exist, so default=None

        xrt.assert_duckarray_equal(ds["shot_point"], shots)
        xrt.assert_duckarray_equal(ds["cable"], cables)

        if chan_header_type == StreamerShotGeometryType.B and grid_override is None:
            expected = list(range(1, np.sum(receivers_per_cable) + 1))
        else:
            expected = list(range(1, np.amax(receivers_per_cable) + 1))
        xrt.assert_duckarray_equal(ds["channel"], expected)

        expected = list(range(0, num_samples, 1))
        xrt.assert_duckarray_equal(ds["time"], expected)


@pytest.mark.parametrize("chan_header_type", [StreamerShotGeometryType.A])
class TestImport4DSparse:
    """Test for 4D segy import with grid overrides."""

    def test_import_4d_segy(  # noqa: PLR0913
        self,
        segy_mock_4d_shots: dict[StreamerShotGeometryType, Path],
        zarr_tmp: Path,
        chan_header_type: StreamerShotGeometryType,
    ) -> None:
        """Test importing a SEG-Y file to MDIO."""
        segy_spec: SegySpec = get_segy_mock_4d_spec()
        segy_path = segy_mock_4d_shots[chan_header_type]
        os.environ["MDIO__GRID__SPARSITY_RATIO_LIMIT"] = "1.1"

        # chunksize=(8, 2, 128, 1024),
        with pytest.raises(GridTraceSparsityError) as execinfo:
            segy_to_mdio(
                segy_spec=segy_spec,
                mdio_template=TemplateRegistry().get("PreStackShotGathers3DTime"),
                input_path=segy_path,
                output_path=zarr_tmp,
                overwrite=True,
            )

        os.environ["MDIO__GRID__SPARSITY_RATIO_LIMIT"] = "10"
        assert "This grid is very sparse and most likely user error with indexing." in str(execinfo.value)


# TODO(Altay): Finish implementing these grid overrides.
# https://github.com/TGSAI/mdio-python/issues/612
@pytest.mark.skip(reason="AutoShotWrap requires a template that is not implemented yet.")
@pytest.mark.parametrize("grid_override", [{"AutoChannelWrap": True}, {"AutoShotWrap": True}, None])
@pytest.mark.parametrize("chan_header_type", [StreamerShotGeometryType.A, StreamerShotGeometryType.B])
class TestImport6D:
    """Test for 6D segy import with grid overrides."""

    def test_import_6d_segy(  # noqa: PLR0913
        self,
        segy_mock_4d_shots: dict[StreamerShotGeometryType, Path],
        zarr_tmp: Path,
        grid_override: dict[str, Any],
        chan_header_type: StreamerShotGeometryType,
    ) -> None:
        """Test importing a SEG-Y file to MDIO."""
        segy_spec: SegySpec = get_segy_mock_4d_spec()
        segy_path = segy_mock_4d_shots[chan_header_type]

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get("XYZ"),  # Placeholder for the template
            input_path=segy_path,
            output_path=zarr_tmp,
            overwrite=True,
            grid_overrides=grid_override,
        )

        # Expected values
        num_samples = 25
        shots = [2, 3, 5, 6, 7, 8, 9]  # original shot list
        if grid_override is not None and "AutoShotWrap" in grid_override:
            shots_new = [int(shot / 2) for shot in shots]  # Updated shot index when ingesting with 2 guns
            shots_set = set(shots_new)  # remove duplicates
            shots = list(shots_set)  # Unique shot points for 6D indexed with gun
        cables = [0, 101, 201, 301]
        guns = [1, 2]
        receivers_per_cable = [1, 5, 7, 5]

        ds = open_mdio(zarr_tmp)

        xrt.assert_duckarray_equal(ds["gun"], guns)
        xrt.assert_duckarray_equal(ds["shot_point"], shots)
        xrt.assert_duckarray_equal(ds["cable"], cables)

        if chan_header_type == StreamerShotGeometryType.B and grid_override is None:
            expected = list(range(1, np.sum(receivers_per_cable) + 1))
        else:
            expected = list(range(1, np.amax(receivers_per_cable) + 1))
        xrt.assert_duckarray_equal(ds["channel"], expected)

        times_expected = list(range(0, num_samples, 1))
        xrt.assert_duckarray_equal(ds["time"], times_expected)


@pytest.mark.dependency
def test_3d_import(segy_input: Path, zarr_tmp: Path) -> None:
    """Test importing a SEG-Y file to MDIO.

    NOTE: This test must be executed before the 'TestReader' and 'TestExport' tests.
    """
    segy_to_mdio(
        segy_spec=custom_teapot_dome_segy_spec(keep_unaltered=True),
        mdio_template=TemplateRegistry().get("PostStack3DTime"),
        input_path=segy_input,
        output_path=zarr_tmp,
        overwrite=True,
    )


@pytest.mark.dependency("test_3d_import")
class TestReader:
    """Test reader functionality.

    NOTE: These tests must be executed after the 'test_3d_import' and before running 'TestExport' tests.
    """

    def test_dataset_metadata(self, zarr_tmp: Path) -> None:
        """Metadata reading tests."""
        ds = open_mdio(zarr_tmp)
        expected_attrs = {
            "apiVersion": __version__,
            "createdOn": "2025-08-06 16:21:54.747880+00:00",
            "name": "PostStack3DTime",
        }
        actual_attrs_json = ds.attrs
        # compare one by one due to ever changing createdOn. For it, we only check existence
        for key, value in expected_attrs.items():
            assert key in actual_attrs_json
            if key == "createdOn":
                assert actual_attrs_json[key] is not None
            else:
                assert actual_attrs_json[key] == value

        attributes = ds.attrs["attributes"]
        assert attributes is not None
        assert len(attributes) == 3
        # Validate all attributes provided by the abstract template
        assert attributes["defaultVariableName"] == "amplitude"
        assert attributes["surveyType"] == "3D"
        assert attributes["gatherType"] == "stacked"

        segy_file_header = ds["segy_file_header"]
        assert segy_file_header.attrs["textHeader"] == text_header_teapot_dome()
        assert segy_file_header.attrs["binaryHeader"] == binary_header_teapot_dome()

    def test_variable_metadata(self, zarr_tmp: Path) -> None:
        """Metadata reading tests."""
        ds = open_mdio(zarr_tmp)
        expected_attrs = {
            "count": 97354860,
            "sum": -8594.551666259766,
            "sumSquares": 40571291.6875,
            "min": -8.375323295593262,
            "max": 7.723702430725098,
            "histogram": {"counts": [], "binCenters": []},
        }
        actual_attrs_json = json.loads(ds["amplitude"].attrs["statsV1"])
        assert actual_attrs_json == expected_attrs

    def test_grid(self, zarr_tmp: Path) -> None:
        """Test validating MDIO variables."""
        ds = open_mdio(zarr_tmp)

        # Validate the dimension coordinate variables
        validate_variable(ds, "inline", (345,), ("inline",), np.int32, range(1, 346), get_values)
        validate_variable(ds, "crossline", (188,), ("crossline",), np.int32, range(1, 189), get_values)
        validate_variable(ds, "time", (1501,), ("time",), np.int32, range(0, 3002, 2), get_values)

        # Validate the non-dimensional coordinate variables
        validate_variable(ds, "cdp_x", (345, 188), ("inline", "crossline"), np.float64, None, None)
        validate_variable(ds, "cdp_y", (345, 188), ("inline", "crossline"), np.float64, None, None)

        # Validate the headers
        # We have a custom set of headers since we used customize_segy_specs()
        segy_spec = custom_teapot_dome_segy_spec(keep_unaltered=True)
        data_type = segy_spec.trace.header.dtype

        validate_variable(
            ds,
            "headers",
            (345, 188),
            ("inline", "crossline"),
            data_type.newbyteorder("native"),  # mdio saves with machine endian, spec could be different endian
            range(1, 346),
            get_inline_header_values,
        )

        # Validate the trace mask
        validate_variable(ds, "trace_mask", (345, 188), ("inline", "crossline"), np.bool, None, None)

        # validate the amplitude data
        validate_variable(
            ds,
            "amplitude",
            (345, 188, 1501),
            ("inline", "crossline", "time"),
            np.float32,
            None,
            None,
        )

    def test_inline_reads(self, zarr_tmp: Path) -> None:
        """Read and compare every 75 inlines' mean and std. dev."""
        ds = open_mdio(zarr_tmp)
        inlines = ds["amplitude"][::75, :, :]
        mean, std = inlines.mean(), inlines.std()
        npt.assert_allclose([mean, std], [1.0555277e-04, 6.0027051e-01])

    def test_crossline_reads(self, zarr_tmp: Path) -> None:
        """Read and compare every 75 crosslines' mean and std. dev."""
        ds = open_mdio(zarr_tmp)
        xlines = ds["amplitude"][:, ::75, :]
        mean, std = xlines.mean(), xlines.std()

        npt.assert_allclose([mean, std], [-5.0329847e-05, 5.9406823e-01])

    def test_zslice_reads(self, zarr_tmp: Path) -> None:
        """Read and compare every 225 z-slices' mean and std. dev."""
        ds = open_mdio(zarr_tmp)
        slices = ds["amplitude"][:, :, ::225]
        mean, std = slices.mean(), slices.std()
        npt.assert_allclose([mean, std], [0.005236923, 0.61279935])


@pytest.mark.dependency("test_3d_import")
class TestExport:
    """Test SEG-Y exporting functionality.

    NOTE: This test(s) must be executed after the 'test_3d_import' and 'TestReader' tests successfully complete.
    """

    def test_3d_export(self, segy_input: Path, zarr_tmp: Path, segy_export_tmp: Path) -> None:
        """Test 3D export to IBM and IEEE."""
        rng = np.random.default_rng(seed=1234)

        spec = custom_teapot_dome_segy_spec(keep_unaltered=True)

        mdio_to_segy(segy_spec=spec, input_path=zarr_tmp, output_path=segy_export_tmp)

        # Check if file sizes match on IBM file.
        assert segy_input.stat().st_size == segy_export_tmp.stat().st_size

        # IBM. Is random original traces and headers match round-trip file?
        in_segy = SegyFile(segy_input, spec=spec)
        out_segy = SegyFile(segy_export_tmp, spec=spec)

        num_traces = in_segy.num_traces
        random_indices = rng.choice(num_traces, 100, replace=False)
        in_traces = in_segy.trace[random_indices]
        out_traces = out_segy.trace[random_indices]

        assert in_segy.num_traces == out_segy.num_traces
        assert in_segy.text_header == out_segy.text_header
        assert in_segy.binary_header == out_segy.binary_header
        # TODO (Dmitriy Repin): Reconcile custom SegySpecs used in the roundtrip SEGY -> MDIO -> SEGY tests
        # https://github.com/TGSAI/mdio-python/issues/610
        npt.assert_array_equal(desired=in_traces.header, actual=out_traces.header)
        npt.assert_array_equal(desired=in_traces.sample, actual=out_traces.sample)
