"""End to end testing for SEG-Y to MDIO conversion and back."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import dask
import numpy as np
import numpy.testing as npt
import pytest
from segy.schema import HeaderField
from segy.schema import ScalarType
from segy.standards import get_segy_standard
from tests.integration.testing_helpers import get_values
from tests.integration.testing_helpers import validate_variable

from mdio import __version__
from mdio import mdio_to_segy
from mdio.api.io import open_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.converters.segy import segy_to_mdio
from mdio.segy.file import SegyFileWrapper

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    import xarray as xr
    from segy.schema import SegySpec


dask.config.set(scheduler="synchronous")


@pytest.fixture
def set_env_vars(monkeypatch: Generator[pytest.MonkeyPatch]) -> None:
    """Set environment variables for the Soda Lake shot tests."""
    monkeypatch.setenv("MDIO__IMPORT__SAVE_SEGY_FILE_HEADER", "true")
    monkeypatch.setenv("MDIO__IMPORT__RAW_HEADERS", "true")


@pytest.fixture
def soda_lake_segy_spec() -> SegySpec:
    """Return the SEG-Y specification for a Soda Lake 2010 raw shot record."""
    shot_fields = [
        HeaderField(name="shot_point", byte=9, format=ScalarType.INT32),
        HeaderField(name="channel", byte=13, format=ScalarType.INT32),
    ]
    return get_segy_standard(1.0).customize(trace_header_fields=shot_fields)


def get_shot_header_values(arr: xr.DataArray) -> np.ndarray:
    """Extract shot_point coordinate values from a headers array."""
    return arr["shot_point"].values


def text_header_soda_lake() -> str:
    """Return the Soda Lake shot expected text header."""
    header_rows = [
        "C 1 CLIENT                        COMPANY                       CREW NO         ",
        "C 2 LINE            AREA                        MAP ID                          ",
        "C 3 REEL NO           DAY-START OF REEL     YEAR      OBSERVER                  ",
        "C 4 INSTRUMENT: MFG            MODEL            SERIAL NO                       ",
        "C 5 DATA TRACES/RECORD        AUXILIARY TRACES/RECORD         CDP FOLD          ",
        "C 6 SAMPLE INTERVAL         SAMPLES/TRACE       BITS/IN      BYTES/SAMPLE       ",
        "C 7 RECORDING FORMAT        FORMAT THIS REEL        MEASUREMENT SYSTEM          ",
        "C 8 SAMPLE CODE: FLOATING PT     FIXED PT     FIXED PT-GAIN     CORRELATED      ",
        "C 9 GAIN  TYPE: FIXED     BINARY     FLOATING POINT     OTHER                   ",
        "C10 FILTERS: ALIAS     HZ  NOTCH     HZ  BAND     -     HZ  SLOPE    -    DB/OCT",
        "C11 SOURCE: TYPE            NUMBER/POINT        POINT INTERVAL                  ",
        "C12     PATTERN:                           LENGTH        WIDTH                  ",
        "C13 SWEEP: START     HZ  END     HZ  LENGTH      MS  CHANNEL NO     TYPE        ",
        "C14 TAPER: START LENGTH       MS  END LENGTH       MS  TYPE                     ",
        "C15 SPREAD: OFFSET        MAX DISTANCE        GROUP INTERVAL                    ",
        "C16 GEOPHONES: PER GROUP     SPACING     FREQUENCY     MFG          MODEL       ",
        "C17     PATTERN:                           LENGTH        WIDTH                  ",
        "C18 TRACES SORTED BY: RECORD     CDP     OTHER                                  ",
        "C19 AMPLITUDE RECOVERY: NONE      SPHERICAL DIV      AGC     OTHER              ",
        "C20 MAP PROJECTION                      ZONE ID       COORDINATE UNITS          ",
        "C21 PROCESSING:                                                                 ",
        "C22 PROCESSING:                                                                 ",
        "C23                                                                             ",
        "C24                                                                             ",
        "C25                                                                             ",
        "C26                                                                             ",
        "C27                                                                             ",
        "C28                                                                             ",
        "C29                                                                             ",
        "C30                                                                             ",
        "C31                                                                             ",
        "C32                                                                             ",
        "C33                                                                             ",
        "C34                                                                             ",
        "C35                                                                             ",
        "C36                                                                             ",
        "C37                                                                             ",
        "C38                                                                             ",
        "C39                                                                             ",
        "C40 END EBCDIC                                                                  ",
    ]
    return "\n".join(header_rows)


def binary_header_soda_lake() -> dict[str, int]:
    """Return the Soda Lake shot expected binary header."""
    return {
        "job_id": 18909,
        "line_num": 1,
        "reel_num": 1,
        "data_traces_per_ensemble": 957,
        "aux_traces_per_ensemble": 1,
        "sample_interval": 2000,
        "orig_sample_interval": 2000,
        "samples_per_trace": 2000,
        "orig_samples_per_trace": 2000,
        "data_sample_format": 1,
        "ensemble_fold": 0,
        "trace_sorting_code": 1,
        "vertical_sum_code": 1,
        "sweep_freq_start": 0,
        "sweep_freq_end": 0,
        "sweep_length": 0,
        "sweep_type_code": 4,
        "sweep_trace_num": 0,
        "sweep_taper_start": 0,
        "sweep_taper_end": 0,
        "taper_type_code": 3,
        "correlated_data_code": 1,
        "binary_gain_code": 2,
        "amp_recovery_code": 1,
        "measurement_system_code": 2,
        "impulse_polarity_code": 0,
        "vibratory_polarity_code": 0,
        "fixed_length_trace_flag": 1,
        "num_extended_text_headers": 0,
        "segy_revision_major": 0,
        "segy_revision_minor": 0,
    }


def raw_binary_header_soda_lake() -> str:
    """Return the Soda Lake shot expected raw binary header, base64 encoded."""
    return (
        "AABJ3QAAAAEAAAABA70AAQfQB9AH0AfQAAEAAAABAAEAAAAAAAAABAAAAAAAAAADAAEAAgABAAIAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAQAAA70AAAfQAAAH0AAAIDAAAHU8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    )


class TestSodaLakeShotRoundtrip:
    """Tests for Soda Lake raw-shot ingestion and export."""

    @pytest.mark.dependency
    @pytest.mark.usefixtures("set_env_vars")
    def test_soda_lake_import(
        self,
        segy_input: Path,
        zarr_tmp: Path,
        soda_lake_segy_spec: SegySpec,
    ) -> None:
        """Test importing a SEG-Y shot record to MDIO."""
        segy_to_mdio(
            segy_spec=soda_lake_segy_spec,
            mdio_template=TemplateRegistry().get("StreamerShotGathers2D"),
            input_path=segy_input,
            output_path=zarr_tmp,
            overwrite=True,
        )

    @pytest.mark.dependency("test_soda_lake_import")
    def test_dataset_metadata(self, zarr_tmp: Path) -> None:
        """Metadata reading tests."""
        ds = open_mdio(zarr_tmp)
        expected_attrs = {
            "apiVersion": __version__,
            "createdOn": "2025-08-06 16:21:54.747880+00:00",
            "name": "StreamerShotGathers2D",
        }
        actual_attrs_json = ds.attrs
        for key, value in expected_attrs.items():
            assert key in actual_attrs_json
            if key == "createdOn":
                assert actual_attrs_json[key] is not None
            else:
                assert actual_attrs_json[key] == value

        attributes = ds.attrs["attributes"]
        assert attributes is not None
        assert attributes["defaultVariableName"] == "amplitude"
        assert attributes["surveyType"] == "2D"
        assert attributes["gatherType"] == "common_source"

        segy_file_header = ds["segy_file_header"]
        assert segy_file_header.attrs["textHeader"] == text_header_soda_lake()
        assert segy_file_header.attrs["binaryHeader"] == binary_header_soda_lake()
        assert segy_file_header.attrs["rawBinaryHeader"] == raw_binary_header_soda_lake()

    def test_variable_metadata(self, zarr_tmp: Path) -> None:
        """Metadata reading tests."""
        ds = open_mdio(zarr_tmp)
        expected_attrs = {
            "count": 1909984,
            "sum": 14.940853472779583,
            "sumSquares": 3365312.882629699,
            "min": -315.4365234375,
            "max": 918.68701171875,
            "histogram": {"counts": [], "binCenters": []},
        }
        actual_attrs = json.loads(ds["amplitude"].attrs["statsV1"])
        assert actual_attrs.keys() == expected_attrs.keys()
        actual_attrs.pop("histogram")
        expected_attrs.pop("histogram")
        np.testing.assert_allclose(list(actual_attrs.values()), list(expected_attrs.values()))

    def test_grid(self, zarr_tmp: Path, soda_lake_segy_spec: SegySpec) -> None:
        """Test validating MDIO variables."""
        ds = open_mdio(zarr_tmp)

        validate_variable(ds, "shot_point", (1,), ("shot_point",), np.int32, [7733], get_values)
        validate_variable(ds, "channel", (958,), ("channel",), np.int32, range(1, 959), get_values)
        validate_variable(ds, "time", (2000,), ("time",), np.int32, range(0, 4000, 2), get_values)

        validate_variable(ds, "source_coord_x", (1,), ("shot_point",), np.float64, None, None)
        validate_variable(ds, "source_coord_y", (1,), ("shot_point",), np.float64, None, None)
        validate_variable(ds, "group_coord_x", (1, 958), ("shot_point", "channel"), np.float64, None, None)
        validate_variable(ds, "group_coord_y", (1, 958), ("shot_point", "channel"), np.float64, None, None)

        data_type = soda_lake_segy_spec.trace.header.dtype
        validate_variable(
            ds,
            "headers",
            (1, 958),
            ("shot_point", "channel"),
            data_type.newbyteorder("native"),
            [7733],
            get_shot_header_values,
        )
        validate_variable(ds, "trace_mask", (1, 958), ("shot_point", "channel"), np.bool, None, None)
        validate_variable(
            ds,
            "amplitude",
            (1, 958, 2000),
            ("shot_point", "channel", "time"),
            np.float32,
            None,
            None,
        )

    def test_shot_reads(self, zarr_tmp: Path) -> None:
        """Read the single shot gather mean and std. dev."""
        ds = open_mdio(zarr_tmp)
        gather = ds["amplitude"][0, :, :]
        mean, std = gather.mean(dtype="float64"), gather.std(dtype="float64")
        npt.assert_allclose([mean, std], [7.79794023008e-06, 1.32530237878])  # 11 precision

    def test_channel_reads(self, zarr_tmp: Path) -> None:
        """Read and compare every 100 channels' mean and std. dev."""
        ds = open_mdio(zarr_tmp)
        channels = ds["amplitude"][:, ::100, :]
        mean, std = channels.mean(dtype="float64"), channels.std(dtype="float64")
        npt.assert_allclose([mean, std], [-3.78133823205e-05, 12.76665926989])  # 11 precision

    def test_zslice_reads(self, zarr_tmp: Path) -> None:
        """Read and compare every 250 z-slices' mean and std. dev."""
        ds = open_mdio(zarr_tmp)
        slices = ds["amplitude"][:, :, ::250]
        mean, std = slices.mean(dtype="float64"), slices.std(dtype="float64")
        npt.assert_allclose([mean, std], [0.11977056380, 10.49600920141])  # 11 precision

    @pytest.mark.dependency("test_soda_lake_import")
    def test_export(
        self, segy_input: Path, zarr_tmp: Path, segy_export_tmp: Path, soda_lake_segy_spec: SegySpec
    ) -> None:
        """Test shot export round-trip."""
        rng = np.random.default_rng(seed=1234)

        mdio_to_segy(segy_spec=soda_lake_segy_spec, input_path=zarr_tmp, output_path=segy_export_tmp)

        assert segy_input.stat().st_size == segy_export_tmp.stat().st_size

        in_segy = SegyFileWrapper(segy_input, spec=soda_lake_segy_spec)
        out_segy = SegyFileWrapper(segy_export_tmp, spec=soda_lake_segy_spec)

        num_traces = in_segy.num_traces
        random_indices = rng.choice(num_traces, 100, replace=False)
        in_traces = in_segy.trace[random_indices]
        out_traces = out_segy.trace[random_indices]

        assert in_segy.num_traces == out_segy.num_traces
        assert in_segy.text_header == out_segy.text_header
        assert in_segy.binary_header == out_segy.binary_header
        npt.assert_array_equal(desired=in_traces.header, actual=out_traces.header)
        # IBM float32 cannot preserve denormals through IEEE decode/encode.
        npt.assert_allclose(desired=in_traces.sample, actual=out_traces.sample, rtol=0, atol=1e-37)
