"""End to end testing for SEG-Y to MDIO conversion and back."""

from os.path import getsize

from typing import Any

import dask
import numpy as np
import numpy.testing as npt
import pytest
from segy import SegyFile

from mdio import MDIOReader
from mdio import mdio_to_segy
from mdio.converters import segy_to_mdio
from mdio.core import Dimension
from mdio.segy.compat import mdio_segy_spec
from mdio.segy.geometry import StreamerShotGeometryType

from pathlib import Path


dask.config.set(scheduler="synchronous")


@pytest.mark.parametrize("index_bytes", [(17, 137)])
@pytest.mark.parametrize("index_names", [("shot_point", "cable")])
@pytest.mark.parametrize("index_types", [("int32", "int16")])
@pytest.mark.parametrize(
    "grid_overrides", [{"NonBinned": True, "chunksize": 2}, {"HasDuplicates": True}]
)
@pytest.mark.parametrize("chan_header_type", [StreamerShotGeometryType.C])
class TestImport4DNonReg:
    """Test for 4D segy import with grid overrides."""

    def test_import_4d_segy(  # noqa: PLR0913
        self,
        segy_mock_4d_shots: dict[StreamerShotGeometryType, str],
        zarr_tmp: str | Path,
        index_bytes: tuple[int, ...],
        index_names: tuple[str, ...],
        index_types: tuple[str, ...],
        grid_overrides: dict[str, Any],
        chan_header_type: StreamerShotGeometryType,
    ) -> None:
        """Test importing a SEG-Y file to MDIO."""
        segy_path = segy_mock_4d_shots[chan_header_type]

        segy_to_mdio(
            segy_path=segy_path,
            mdio_path_or_buffer=str(zarr_tmp),
            index_bytes=index_bytes,
            index_names=index_names,
            index_types=index_types,
            chunksize=(8, 2, 10),
            overwrite=True,
            grid_overrides=grid_overrides,
        )

        # Expected values
        num_samples = 25
        shots = [2, 3, 5, 6, 7, 8, 9]
        cables = [0, 101, 201, 301]
        receivers_per_cable = [1, 5, 7, 5]

        # QC mdio output
        mdio = MDIOReader(str(zarr_tmp), access_pattern="0123")
        assert mdio.binary_header["samples_per_trace"] == num_samples
        grid = mdio.grid

        assert grid.select_dim(index_names[0]) == Dimension(shots, index_names[0])
        assert grid.select_dim(index_names[1]) == Dimension(cables, index_names[1])
        assert grid.select_dim("trace") == Dimension(
            range(1, np.amax(receivers_per_cable) + 1), "trace"
        )
        samples_exp = Dimension(range(0, num_samples, 1), "sample")
        assert grid.select_dim("sample") == samples_exp


@pytest.mark.parametrize("index_bytes", [(17, 137, 13)])
@pytest.mark.parametrize("index_names", [("shot_point", "cable", "channel")])
@pytest.mark.parametrize("index_types", [("int32", "int16", "int32")])
@pytest.mark.parametrize("grid_overrides", [{"AutoChannelWrap": True}, None])
@pytest.mark.parametrize(
    "chan_header_type", [StreamerShotGeometryType.A, StreamerShotGeometryType.B]
)
class TestImport4D:
    """Test for 4D segy import with grid overrides."""

    def test_import_4d_segy(  # noqa: PLR0913
        self,
        segy_mock_4d_shots: dict[StreamerShotGeometryType, str],
        zarr_tmp: str | Path,
        index_bytes: tuple[int, ...],
        index_names: tuple[str, ...],
        index_types: tuple[str, ...],
        grid_overrides: dict[str, Any],
        chan_header_type: StreamerShotGeometryType,
    ) -> None:
        """Test importing a SEG-Y file to MDIO."""
        segy_path = segy_mock_4d_shots[chan_header_type]

        segy_to_mdio(
            segy_path=segy_path,
            mdio_path_or_buffer=str(zarr_tmp),
            index_bytes=index_bytes,
            index_names=index_names,
            index_types=index_types,
            chunksize=(8, 2, 128, 1024),
            overwrite=True,
            grid_overrides=grid_overrides,
        )

        # Expected values
        num_samples = 25
        shots = [2, 3, 5, 6, 7, 8, 9]
        cables = [0, 101, 201, 301]
        receivers_per_cable = [1, 5, 7, 5]

        # QC mdio output
        mdio = MDIOReader(str(zarr_tmp), access_pattern="0123")
        assert mdio.binary_header["samples_per_trace"] == num_samples
        grid = mdio.grid

        assert grid.select_dim(index_names[0]) == Dimension(shots, index_names[0])
        assert grid.select_dim(index_names[1]) == Dimension(cables, index_names[1])

        if chan_header_type == StreamerShotGeometryType.B and grid_overrides is None:
            assert grid.select_dim(index_names[2]) == Dimension(
                range(1, np.sum(receivers_per_cable) + 1), index_names[2]
            )
        else:
            assert grid.select_dim(index_names[2]) == Dimension(
                range(1, np.amax(receivers_per_cable) + 1), index_names[2]
            )

        samples_exp = Dimension(range(0, num_samples, 1), "sample")
        assert grid.select_dim("sample") == samples_exp


@pytest.mark.parametrize("index_bytes", [(17, 137, 13)])
@pytest.mark.parametrize("index_names", [("shot_point", "cable", "channel")])
@pytest.mark.parametrize("index_types", [("int32", "int16", "int32")])
@pytest.mark.parametrize("grid_overrides", [None])
@pytest.mark.parametrize("chan_header_type", [StreamerShotGeometryType.A])
class TestImport4DSparse:
    """Test for 4D segy import with grid overrides."""

    def test_import_4d_segy(  # noqa: PLR0913
        self,
        segy_mock_4d_shots: dict[StreamerShotGeometryType, str],
        zarr_tmp: str | Path,
        index_bytes: tuple[int, ...],
        index_names: tuple[str, ...],
        index_types: tuple[str, ...],
        grid_overrides: dict[str, Any] | None,
        chan_header_type: StreamerShotGeometryType,
    ) -> None:
        """Test importing a SEG-Y file to MDIO."""
        import os

        from mdio.converters.exceptions import GridTraceSparsityError

        segy_path = segy_mock_4d_shots[chan_header_type]
        os.environ["MDIO__GRID__SPARSITY_RATIO_LIMIT"] = "1.1"

        with pytest.raises(GridTraceSparsityError) as execinfo:
            segy_to_mdio(
                segy_path=segy_path,
                mdio_path_or_buffer=str(zarr_tmp),
                index_bytes=index_bytes,
                index_names=index_names,
                index_types=index_types,
                chunksize=(8, 2, 128, 1024),
                overwrite=True,
                grid_overrides=grid_overrides,
            )

        os.environ["MDIO__GRID__SPARSITY_RATIO_LIMIT"] = "10"
        assert (
            "This grid is very sparse and most likely user error with indexing."
            in str(execinfo.value)
        )


@pytest.mark.parametrize("index_bytes", [(133, 171, 17, 137, 13)])
@pytest.mark.parametrize(
    "index_names", [("shot_line", "gun", "shot_point", "cable", "channel")]
)
@pytest.mark.parametrize("index_types", [("int16", "int16", "int32", "int16", "int32")])
@pytest.mark.parametrize(
    "grid_overrides", [{"AutoChannelWrap": True, "AutoShotWrap": True}, None]
)
@pytest.mark.parametrize(
    "chan_header_type", [StreamerShotGeometryType.A, StreamerShotGeometryType.B]
)
class TestImport6D:
    """Test for 6D segy import with grid overrides."""

    def test_import_6d_segy(  # noqa: PLR0913
        self,
        segy_mock_4d_shots: dict[StreamerShotGeometryType, str],
        zarr_tmp: str | Path,
        index_bytes: tuple[int, ...],
        index_names: tuple[str, ...],
        index_types: tuple[str, ...],
        grid_overrides: dict[str, Any] | None,
        chan_header_type: StreamerShotGeometryType,
    ) -> None:
        """Test importing a SEG-Y file to MDIO."""
        segy_path = segy_mock_4d_shots[chan_header_type]

        segy_to_mdio(
            segy_path=segy_path,
            mdio_path_or_buffer=str(zarr_tmp),
            index_bytes=index_bytes,
            index_names=index_names,
            index_types=index_types,
            chunksize=(1, 1, 8, 1, 12, 36),
            overwrite=True,
            grid_overrides=grid_overrides,
        )

        # Expected values
        num_samples = 25
        shots = [2, 3, 5, 6, 7, 8, 9]  # original shot list
        if grid_overrides is not None and "AutoShotWrap" in grid_overrides:
            shots_new = [
                int(shot / 2) for shot in shots
            ]  # Updated shot index when ingesting with 2 guns
            shots_set = set(shots_new)  # remove duplicates
            shots = list(shots_set)  # Unique shot points for 6D indexed with gun
        cables = [0, 101, 201, 301]
        guns = [1, 2]
        receivers_per_cable = [1, 5, 7, 5]

        # QC mdio output
        mdio = MDIOReader(str(zarr_tmp), access_pattern="012345")
        assert mdio.binary_header["samples_per_trace"] == num_samples
        grid = mdio.grid

        assert grid.select_dim(index_names[1]) == Dimension(guns, index_names[1])
        assert grid.select_dim(index_names[2]) == Dimension(shots, index_names[2])
        assert grid.select_dim(index_names[3]) == Dimension(cables, index_names[3])

        if chan_header_type == StreamerShotGeometryType.B and grid_overrides is None:
            assert grid.select_dim(index_names[4]) == Dimension(
                range(1, np.sum(receivers_per_cable) + 1), index_names[4]
            )
        else:
            assert grid.select_dim(index_names[4]) == Dimension(
                range(1, np.amax(receivers_per_cable) + 1), index_names[4]
            )

        samples_exp = Dimension(range(0, num_samples, 1), "sample")
        assert grid.select_dim("sample") == samples_exp


@pytest.mark.dependency
@pytest.mark.parametrize("index_bytes", [(17, 13)])
@pytest.mark.parametrize("index_names", [("inline", "crossline")])
def test_3d_import(segy_input, zarr_tmp, index_bytes, index_names):
    """Test importing a SEG-Y file to MDIO."""
    segy_to_mdio(
        segy_path=segy_input.__str__(),
        mdio_path_or_buffer=zarr_tmp.__str__(),
        index_bytes=index_bytes,
        index_names=index_names,
        overwrite=True,
    )


@pytest.mark.dependency("test_3d_import")
class TestReader:
    """Test reader functionality."""

    def test_meta_read(self, zarr_tmp: Path) -> None:
        """Metadata reading tests."""
        mdio = MDIOReader(str(zarr_tmp))
        assert mdio.binary_header["samples_per_trace"] == 1501
        assert mdio.binary_header["sample_interval"] == 2000

    def test_grid(self, zarr_tmp: Path) -> None:
        """Grid reading tests."""
        mdio = MDIOReader(str(zarr_tmp))
        grid = mdio.grid

        assert grid.select_dim("inline") == Dimension(range(1, 346), "inline")
        assert grid.select_dim("crossline") == Dimension(range(1, 189), "crossline")
        assert grid.select_dim("sample") == Dimension(range(0, 3002, 2), "sample")

    def test_get_data(self, zarr_tmp: Path) -> None:
        """Data retrieval tests."""
        mdio = MDIOReader(str(zarr_tmp))

        assert mdio.shape == (345, 188, 1501)
        assert mdio[0, :, :].shape == (188, 1501)
        assert mdio[:, 0, :].shape == (345, 1501)
        assert mdio[:, :, 0].shape == (345, 188)

    def test_inline(self, zarr_tmp: Path) -> None:
        """Read and compare every 75 inlines' mean and std. dev."""
        mdio = MDIOReader(str(zarr_tmp))

        inlines = mdio[::75, :, :]
        mean, std = inlines.mean(), inlines.std()

        npt.assert_allclose([mean, std], [1.0555277e-04, 6.0027051e-01])

    def test_crossline(self, zarr_tmp: Path) -> None:
        """Read and compare every 75 crosslines' mean and std. dev."""
        mdio = MDIOReader(str(zarr_tmp))

        xlines = mdio[:, ::75, :]
        mean, std = xlines.mean(), xlines.std()

        npt.assert_allclose([mean, std], [-5.0329847e-05, 5.9406823e-01])

    def test_zslice(self, zarr_tmp: Path) -> None:
        """Read and compare every 225 z-slices' mean and std. dev."""
        mdio = MDIOReader(str(zarr_tmp))

        slices = mdio[:, :, ::225]
        mean, std = slices.mean(), slices.std()

        npt.assert_allclose([mean, std], [0.005236923, 0.61279935])


@pytest.mark.dependency("test_3d_import")
class TestExport:
    """Test SEG-Y exporting functionaliy."""

    def test_3d_export(self, zarr_tmp: Path, segy_export_tmp: Path) -> None:
        """Test 3D export to IBM and IEEE."""
        mdio_to_segy(
            mdio_path_or_buffer=str(zarr_tmp),
            output_segy_path=str(segy_export_tmp),
        )

    def test_size_equal(self, segy_input: Path, segy_export_tmp: Path) -> None:
        """Check if file sizes match on IBM file."""
        assert getsize(segy_input) == getsize(segy_export_tmp)

    def test_rand_equal(self, segy_input: Path, segy_export_tmp: Path) -> None:
        """IBM. Is random original traces and headers match round-trip file?"""
        spec = mdio_segy_spec()

        in_segy = SegyFile(segy_input, spec=spec)
        out_segy = SegyFile(segy_export_tmp, spec=spec)

        num_traces = in_segy.num_traces
        random_indices = np.random.choice(num_traces, 100, replace=False)
        in_traces = in_segy.trace[random_indices]
        out_traces = out_segy.trace[random_indices]

        assert in_segy.num_traces == out_segy.num_traces
        assert in_segy.text_header == out_segy.text_header
        assert in_segy.binary_header == out_segy.binary_header
        npt.assert_array_equal(in_traces.header, out_traces.header)
        npt.assert_array_equal(in_traces.sample, out_traces.sample)
