"""End to end testing for SEG-Y to MDIO conversion and back."""

from os.path import getsize

import dask
import numpy as np
import numpy.testing as npt
import pytest
from segy import SegyFile
from segy.schema import ScalarType
from segy.schema import StructuredFieldDescriptor
from segy.standards import SegyStandard
from segy.standards import rev1_segy

from mdio import MDIOReader
from mdio import mdio_to_segy
from mdio.converters import segy_to_mdio
from mdio.core import Dimension


dask.config.set(scheduler="synchronous")


Int32 = ScalarType.INT32
Int16 = ScalarType.INT16
#
#
# @pytest.mark.parametrize("header_names", [("shot_point", "cable")])
# @pytest.mark.parametrize(
#     "grid_overrides", [{"NonBinned": True, "chunksize": 2}, {"HasDuplicates": True}]
# )
# @pytest.mark.parametrize("chan_header_type", [StreamerShotGeometryType.C])
# class TestImport4DNonReg:
#     """Test for 4D segy import with grid overrides."""
#
#     def test_import_4d_segy(
#         self,
#         segy_mock_4d_shots,
#         zarr_tmp,
#         header_names,
#         grid_overrides,
#         chan_header_type,
#     ):
#         """Test importing a SEG-Y file to MDIO."""
#         segy_path = segy_mock_4d_shots[chan_header_type]
#
#         trace_header_fields = [
#             StructuredFieldDescriptor(name="shot_point", offset=16, format=Int32),
#             StructuredFieldDescriptor(name="cable", offset=136, format=Int16),
#         ]
#
#         segy_spec = rev1_segy.customize(trace_header_fields=trace_header_fields)
#         segy_to_mdio(
#             segy_path=segy_path,
#             mdio_path_or_buffer=zarr_tmp.__str__(),
#             index_names=header_names,
#             chunksize=[8, 2, 10],
#             overwrite=True,
#             grid_overrides=grid_overrides,
#             segy_kwargs={"spec": segy_spec},
#         )
#
#         # Expected values
#         num_samples = 25
#         shots = [2, 3, 5, 6, 7, 8, 9]
#         cables = [0, 101, 201, 301]
#         receivers_per_cable = [1, 5, 7, 5]
#
#         # QC mdio output
#         mdio = MDIOReader(zarr_tmp.__str__(), access_pattern="0123")
#         assert mdio.binary_header["samples_per_trace"] == num_samples
#         grid = mdio.grid
#
#         assert grid.select_dim(header_names[0]) == Dimension(shots, header_names[0])
#         assert grid.select_dim(header_names[1]) == Dimension(cables, header_names[1])
#         assert grid.select_dim("trace") == Dimension(
#             range(1, np.amax(receivers_per_cable) + 1), "trace"
#         )
#         samples_exp = Dimension(range(0, num_samples, 1), "sample")
#         assert grid.select_dim("sample") == samples_exp
#
#
# @pytest.mark.parametrize("header_names", [("shot_point", "cable", "channel")])
# @pytest.mark.parametrize("grid_overrides", [{"AutoChannelWrap": True}, None])
# @pytest.mark.parametrize(
#     "chan_header_type", [StreamerShotGeometryType.A, StreamerShotGeometryType.B]
# )
# class TestImport4D:
#     """Test for 4D segy import with grid overrides."""
#
#     def test_import_4d_segy(
#         self,
#         segy_mock_4d_shots,
#         zarr_tmp,
#         header_names,
#         grid_overrides,
#         chan_header_type,
#     ):
#         """Test importing a SEG-Y file to MDIO."""
#         segy_path = segy_mock_4d_shots[chan_header_type]
#
#         trace_header_fields = [
#             StructuredFieldDescriptor(name="channel", offset=12, format=Int32),
#             StructuredFieldDescriptor(name="shot_point", offset=16, format=Int32),
#             StructuredFieldDescriptor(name="cable", offset=136, format=Int16),
#         ]
#
#         segy_spec = rev1_segy.customize(trace_header_fields=trace_header_fields)
#         segy_to_mdio(
#             segy_path=segy_path,
#             mdio_path_or_buffer=zarr_tmp.__str__(),
#             index_names=header_names,
#             chunksize=[8, 2, 128, 1024],
#             overwrite=True,
#             grid_overrides=grid_overrides,
#             segy_kwargs={"spec": segy_spec},
#         )
#
#         # Expected values
#         num_samples = 25
#         shots = [2, 3, 5, 6, 7, 8, 9]
#         cables = [0, 101, 201, 301]
#         receivers_per_cable = [1, 5, 7, 5]
#
#         # QC mdio output
#         mdio = MDIOReader(zarr_tmp.__str__(), access_pattern="0123")
#         assert mdio.binary_header["samples_per_trace"] == num_samples
#         grid = mdio.grid
#
#         assert grid.select_dim(header_names[0]) == Dimension(shots, header_names[0])
#         assert grid.select_dim(header_names[1]) == Dimension(cables, header_names[1])
#
#         if chan_header_type == StreamerShotGeometryType.B and grid_overrides is None:
#             assert grid.select_dim(header_names[2]) == Dimension(
#                 range(1, np.sum(receivers_per_cable) + 1), header_names[2]
#             )
#         else:
#             assert grid.select_dim(header_names[2]) == Dimension(
#                 range(1, np.amax(receivers_per_cable) + 1), header_names[2]
#             )
#
#         samples_exp = Dimension(range(0, num_samples, 1), "sample")
#         assert grid.select_dim("sample") == samples_exp
#
#
# @pytest.mark.parametrize("header_names", [("shot_point", "cable", "channel")])
# @pytest.mark.parametrize("grid_overrides", [None])
# @pytest.mark.parametrize("chan_header_type", [StreamerShotGeometryType.A])
# class TestImport4DSparse:
#     """Test for 4D segy import with grid overrides."""
#
#     def test_import_4d_segy(
#         self,
#         segy_mock_4d_shots,
#         zarr_tmp,
#         header_names,
#         grid_overrides,
#         chan_header_type,
#     ):
#         """Test importing a SEG-Y file to MDIO."""
#         import os
#
#         from mdio.converters.exceptions import GridTraceSparsityError
#
#         segy_path = segy_mock_4d_shots[chan_header_type]
#         os.environ["MDIO__GRID__SPARSITY_RATIO_LIMIT"] = "1.1"
#
#         trace_header_fields = [
#             StructuredFieldDescriptor(name="channel", offset=12, format=Int32),
#             StructuredFieldDescriptor(name="shot_point", offset=16, format=Int32),
#             StructuredFieldDescriptor(name="cable", offset=136, format=Int16),
#         ]
#         segy_spec = rev1_segy.customize(trace_header_fields=trace_header_fields)
#
#         with pytest.raises(GridTraceSparsityError) as execinfo:
#             segy_to_mdio(
#                 segy_path=segy_path,
#                 mdio_path_or_buffer=zarr_tmp.__str__(),
#                 index_names=header_names,
#                 chunksize=[8, 2, 128, 1024],
#                 overwrite=True,
#                 grid_overrides=grid_overrides,
#                 segy_kwargs={"spec": segy_spec},
#             )
#
#         os.environ["MDIO__GRID__SPARSITY_RATIO_LIMIT"] = "10"
#         assert (
#             "This grid is very sparse and most likely user error with indexing."
#             in str(execinfo.value)
#         )
#
#
# @pytest.mark.parametrize(
#     "header_names", [("shot_line", "gun", "shot_point", "cable", "channel")]
# )
# @pytest.mark.parametrize(
#     "grid_overrides", [{"AutoChannelWrap": True, "AutoShotWrap": True}, None]
# )
# @pytest.mark.parametrize(
#     "chan_header_type", [StreamerShotGeometryType.A, StreamerShotGeometryType.B]
# )
# class TestImport6D:
#     """Test for 6D segy import with grid overrides."""
#
#     def test_import_6d_segy(
#         self,
#         segy_mock_4d_shots,
#         zarr_tmp,
#         header_names,
#         grid_overrides,
#         chan_header_type,
#     ):
#         """Test importing a SEG-Y file to MDIO."""
#         segy_path = segy_mock_4d_shots[chan_header_type]
#
#         trace_header_fields = [
#             StructuredFieldDescriptor(name="channel", offset=12, format=Int32),
#             StructuredFieldDescriptor(name="shot_point", offset=16, format=Int32),
#             StructuredFieldDescriptor(name="shot_line", offset=132, format=Int16),
#             StructuredFieldDescriptor(name="cable", offset=136, format=Int16),
#             StructuredFieldDescriptor(name="gun", offset=170, format=Int16),
#         ]
#
#         segy_spec = rev1_segy.customize(trace_header_fields=trace_header_fields)
#         segy_to_mdio(
#             segy_path=segy_path,
#             mdio_path_or_buffer=zarr_tmp.__str__(),
#             index_names=header_names,
#             chunksize=[1, 1, 8, 1, 12, 36],
#             overwrite=True,
#             grid_overrides=grid_overrides,
#             segy_kwargs={"spec": segy_spec},
#         )
#
#         # Expected values
#         num_samples = 25
#         shots = [2, 3, 5, 6, 7, 8, 9]  # original shot list
#         if grid_overrides is not None and "AutoShotWrap" in grid_overrides:
#             shots_new = [
#                 int(shot / 2) for shot in shots
#             ]  # Updated shot index when ingesting with 2 guns
#             shots_set = set(shots_new)  # remove duplicates
#             shots = list(shots_set)  # Unique shot points for 6D indexed with gun
#         cables = [0, 101, 201, 301]
#         guns = [1, 2]
#         receivers_per_cable = [1, 5, 7, 5]
#
#         # QC mdio output
#         mdio = MDIOReader(zarr_tmp.__str__(), access_pattern="012345")
#         assert mdio.binary_header["samples_per_trace"] == num_samples
#         grid = mdio.grid
#
#         assert grid.select_dim(header_names[1]) == Dimension(guns, header_names[1])
#         assert grid.select_dim(header_names[2]) == Dimension(shots, header_names[2])
#         assert grid.select_dim(header_names[3]) == Dimension(cables, header_names[3])
#
#         if chan_header_type == StreamerShotGeometryType.B and grid_overrides is None:
#             assert grid.select_dim(header_names[4]) == Dimension(
#                 range(1, np.sum(receivers_per_cable) + 1), header_names[4]
#             )
#         else:
#             assert grid.select_dim(header_names[4]) == Dimension(
#                 range(1, np.amax(receivers_per_cable) + 1), header_names[4]
#             )
#
#         samples_exp = Dimension(range(0, num_samples, 1), "sample")
#         assert grid.select_dim("sample") == samples_exp
#


@pytest.mark.parametrize("header_names", [("inline", "crossline")])
class TestImport:
    """Import tests."""

    def test_3d_import(self, segy_input, zarr_tmp, header_names):
        """Test importing a SEG-Y file to MDIO."""
        segy_spec = rev1_segy.customize(
            trace_header_fields=[
                StructuredFieldDescriptor(name="inline", offset=180, format=Int32),
                StructuredFieldDescriptor(name="crossline", offset=184, format=Int32),
            ]
        )
        segy_to_mdio(
            segy_path=segy_input.__str__(),
            mdio_path_or_buffer=zarr_tmp.__str__(),
            index_names=header_names,
            overwrite=True,
            segy_kwargs={"spec": segy_spec},
        )


class TestReader:
    """Test reader functionality."""

    def test_meta_read(self, zarr_tmp):
        """Metadata reading tests."""
        mdio = MDIOReader(zarr_tmp.__str__())
        assert mdio.binary_header["samples_per_trace"] == 1501
        assert mdio.binary_header["sample_interval"] == 2000

    def test_grid(self, zarr_tmp):
        """Grid reading tests."""
        mdio = MDIOReader(zarr_tmp.__str__())
        grid = mdio.grid

        assert grid.select_dim("inline") == Dimension(range(1, 346), "inline")
        assert grid.select_dim("crossline") == Dimension(range(1, 189), "crossline")
        assert grid.select_dim("sample") == Dimension(range(0, 3002, 2), "sample")

    def test_get_data(self, zarr_tmp):
        """Data retrieval tests."""
        mdio = MDIOReader(zarr_tmp.__str__())

        assert mdio.shape == (345, 188, 1501)
        assert mdio[0, :, :].shape == (188, 1501)
        assert mdio[:, 0, :].shape == (345, 1501)
        assert mdio[:, :, 0].shape == (345, 188)

    def test_inline(self, zarr_tmp):
        """Read and compare every 75 inlines' mean and std. dev."""
        mdio = MDIOReader(zarr_tmp.__str__())

        inlines = mdio[::75, :, :]
        mean, std = inlines.mean(), inlines.std()

        npt.assert_allclose([mean, std], [1.0555277e-04, 6.0027051e-01])

    def test_crossline(self, zarr_tmp):
        """Read and compare every 75 crosslines' mean and std. dev."""
        mdio = MDIOReader(zarr_tmp.__str__())

        xlines = mdio[:, ::75, :]
        mean, std = xlines.mean(), xlines.std()

        npt.assert_allclose([mean, std], [-5.0329847e-05, 5.9406823e-01])

    def test_zslice(self, zarr_tmp):
        """Read and compare every 225 z-slices' mean and std. dev."""
        mdio = MDIOReader(zarr_tmp.__str__())

        slices = mdio[:, :, ::225]
        mean, std = slices.mean(), slices.std()

        npt.assert_allclose([mean, std], [0.005236923, 0.61279935])


class TestExport:
    """Test SEG-Y exporting functionaliy."""

    def test_3d_export(self, zarr_tmp, segy_export_tmp):
        """Test 3D export to IBM and IEEE."""
        segy_spec = rev1_segy.customize(
            trace_header_fields=[
                StructuredFieldDescriptor(name="inline", offset=180, format=Int32),
                StructuredFieldDescriptor(name="crossline", offset=184, format=Int32),
            ]
        )
        segy_spec.segy_standard = SegyStandard.REV0

        mdio_to_segy(
            mdio_path_or_buffer=zarr_tmp.__str__(),
            output_segy_path=segy_export_tmp.__str__(),
            segy_kwargs={"spec": segy_spec},
        )

    def test_size_equal(self, segy_input, segy_export_tmp):
        """Check if file sizes match on IBM file."""
        assert getsize(segy_input) == getsize(segy_export_tmp)

    def test_rand_equal(self, segy_input, segy_export_tmp):
        """IBM. Is random original traces and headers match round-trip file?"""
        segy_spec = rev1_segy.customize(
            trace_header_fields=[
                StructuredFieldDescriptor(name="inline", offset=180, format=Int32),
                StructuredFieldDescriptor(name="crossline", offset=184, format=Int32),
            ]
        )

        in_segy = SegyFile(segy_input, spec=segy_spec)
        out_segy = SegyFile(segy_export_tmp, spec=segy_spec)

        num_traces = in_segy.num_traces
        random_indices = list(np.random.randint(0, num_traces, 100))

        in_traces = in_segy.trace[random_indices]
        out_traces = out_segy.trace[random_indices]

        print(in_segy.binary_header.item(), out_segy.binary_header.item())

        assert in_segy.num_traces == out_segy.num_traces
        assert in_segy.text_header == out_segy.text_header
        assert in_segy.binary_header == out_segy.binary_header
        assert in_traces.header == out_traces.header
        npt.assert_array_equal(in_traces.sample, out_traces.sample)
