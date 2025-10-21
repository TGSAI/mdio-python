"""End to end testing for SEG-Y to MDIO conversion and back."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import dask
import numpy as np
import numpy.testing as npt
import pytest
import xarray.testing as xrt
from tests.integration.conftest import get_segy_mock_4d_spec

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
os.environ["MDIO__IMPORT__SAVE_SEGY_FILE_HEADER"] = "true"


# TODO(Altay): Finish implementing these grid overrides.
# https://github.com/TGSAI/mdio-python/issues/612
@pytest.mark.skip(reason="NonBinned and HasDuplicates haven't been properly implemented yet.")
@pytest.mark.parametrize(
    "grid_override", [{"NonBinned": True}, {"HasDuplicates": True}], ids=["NonBinned", "HasDuplicates"]
)
@pytest.mark.parametrize("chan_header_type", [StreamerShotGeometryType.C])
class TestImport4DNonReg:  # pragma: no cover - tests is skipped
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


@pytest.mark.parametrize("grid_override", [{"AutoChannelWrap": True}, None], ids=["AutoChannelWrap", "None"])
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


@pytest.mark.parametrize(
    "grid_override", [{"AutoChannelWrap": True, "AutoShotWrap": True}, None], ids=["Channel&ShotWrap", "None"]
)
@pytest.mark.parametrize("chan_header_type", [StreamerShotGeometryType.A, StreamerShotGeometryType.B])
class TestImport6D:  # pragma: no cover - tests is skipped
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
            mdio_template=TemplateRegistry().get("PreStackGathers3DTime"),  # Placeholder for the template
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
