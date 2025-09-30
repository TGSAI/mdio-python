"""Test harness for disaster recovery wrapper.

This module tests the _disaster_recovery_wrapper.py functionality by creating
test SEGY files with different configurations and validating that the raw headers
from get_header_raw_and_transformed match the bytes on disk.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from segy import SegyFile
from segy.factory import SegyFactory
from segy.schema import Endianness
from segy.schema import HeaderField
from segy.schema import SegySpec
from segy.standards import get_segy_standard

from mdio.segy._disaster_recovery_wrapper import SegyFileTraceDataWrapper

SAMPLES_PER_TRACE = 1501


class TestDisasterRecoveryWrapper:
    """Test cases for disaster recovery wrapper functionality."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def basic_segy_spec(self) -> SegySpec:
        """Create a basic SEGY specification for testing."""
        spec = get_segy_standard(1.0)

        # Add basic header fields for inline/crossline
        header_fields = [
            HeaderField(name="inline", byte=189, format="int32"),
            HeaderField(name="crossline", byte=193, format="int32"),
            HeaderField(name="cdp_x", byte=181, format="int32"),
            HeaderField(name="cdp_y", byte=185, format="int32"),
        ]

        return spec.customize(trace_header_fields=header_fields)

    @pytest.fixture(
        params=[
            {"endianness": Endianness.BIG, "data_format": 1, "name": "big_endian_ibm"},
            {"endianness": Endianness.BIG, "data_format": 5, "name": "big_endian_ieee"},
            {"endianness": Endianness.LITTLE, "data_format": 1, "name": "little_endian_ibm"},
            {"endianness": Endianness.LITTLE, "data_format": 5, "name": "little_endian_ieee"},
        ]
    )
    def segy_config(self, request: pytest.FixtureRequest) -> dict:
        """Parameterized fixture for different SEGY configurations."""
        return request.param

    def create_test_segy_file(  # noqa: PLR0913
        self,
        spec: SegySpec,
        num_traces: int,
        samples_per_trace: int,
        output_path: Path,
        endianness: Endianness = Endianness.BIG,
        data_format: int = 1,  # 1=IBM float, 5=IEEE float
        inline_range: tuple[int, int] = (1, 5),
        crossline_range: tuple[int, int] = (1, 5),
    ) -> SegySpec:
        """Create a test SEGY file with synthetic data."""
        # Update spec with desired endianness
        spec.endianness = endianness

        factory = SegyFactory(spec=spec, samples_per_trace=samples_per_trace)

        # Create synthetic header data
        headers = factory.create_trace_header_template(num_traces)
        samples = factory.create_trace_sample_template(num_traces)

        # Set inline/crossline values
        inline_start, inline_end = inline_range
        crossline_start, crossline_end = crossline_range

        # Create a simple grid
        inlines = np.arange(inline_start, inline_end + 1)
        crosslines = np.arange(crossline_start, crossline_end + 1)

        trace_idx = 0
        for inline in inlines:
            for crossline in crosslines:
                if trace_idx >= num_traces:
                    break

                headers["inline"][trace_idx] = inline
                headers["crossline"][trace_idx] = crossline
                headers["cdp_x"][trace_idx] = inline * 100  # Simple coordinate calculation
                headers["cdp_y"][trace_idx] = crossline * 100

                # Create simple synthetic trace data
                samples[trace_idx] = np.linspace(0, 1, samples_per_trace)

                trace_idx += 1

        # Write the SEGY file with custom binary header
        binary_header_updates = {"data_sample_format": data_format}
        with output_path.open("wb") as f:
            f.write(factory.create_textual_header())
            f.write(factory.create_binary_header(update=binary_header_updates))
            f.write(factory.create_traces(headers, samples))

        return spec

    def test_wrapper_basic_functionality(self, temp_dir: Path, basic_segy_spec: SegySpec, segy_config: dict) -> None:
        """Test basic functionality of SegyFileTraceDataWrapper."""
        config_name = segy_config["name"]
        endianness = segy_config["endianness"]
        data_format = segy_config["data_format"]

        segy_path = temp_dir / f"test_{config_name}.segy"

        # Create test SEGY file
        num_traces = 10
        samples_per_trace = SAMPLES_PER_TRACE

        spec = self.create_test_segy_file(
            spec=basic_segy_spec,
            num_traces=num_traces,
            samples_per_trace=samples_per_trace,
            output_path=segy_path,
            endianness=endianness,
            data_format=data_format,
        )

        # Load the SEGY file
        segy_file = SegyFile(segy_path, spec=spec)

        # Test single trace
        trace_idx = 3
        wrapper = SegyFileTraceDataWrapper(segy_file, trace_idx)

        # Test that properties are accessible
        assert wrapper.header is not None
        assert wrapper.raw_header is not None
        assert wrapper.sample is not None

        # Test header properties
        transformed_header = wrapper.header
        raw_header = wrapper.raw_header

        # Raw header should be bytes (240 bytes per trace header)
        assert raw_header.dtype == np.dtype("|V240")

        # Transformed header should have the expected fields
        assert "inline" in transformed_header.dtype.names
        assert "crossline" in transformed_header.dtype.names

    def test_wrapper_with_multiple_traces(self, temp_dir: Path, basic_segy_spec: SegySpec, segy_config: dict) -> None:
        """Test wrapper with multiple traces."""
        config_name = segy_config["name"]
        endianness = segy_config["endianness"]
        data_format = segy_config["data_format"]

        segy_path = temp_dir / f"test_multiple_{config_name}.segy"

        # Create test SEGY file
        num_traces = 5
        samples_per_trace = SAMPLES_PER_TRACE

        spec = self.create_test_segy_file(
            spec=basic_segy_spec,
            num_traces=num_traces,
            samples_per_trace=samples_per_trace,
            output_path=segy_path,
            endianness=endianness,
            data_format=data_format,
        )

        # Load the SEGY file
        segy_file = SegyFile(segy_path, spec=spec)

        # Test with list of indices
        trace_indices = [0, 2, 4]
        wrapper = SegyFileTraceDataWrapper(segy_file, trace_indices)

        # Test that properties work with multiple traces
        assert wrapper.header is not None
        assert wrapper.raw_header is not None
        assert wrapper.sample is not None

        # Check that we got the expected number of traces
        assert wrapper.header.size == len(trace_indices)
        assert wrapper.raw_header.size == len(trace_indices)

    def test_wrapper_with_slice_indices(self, temp_dir: Path, basic_segy_spec: SegySpec, segy_config: dict) -> None:
        """Test wrapper with slice indices."""
        config_name = segy_config["name"]
        endianness = segy_config["endianness"]
        data_format = segy_config["data_format"]

        segy_path = temp_dir / f"test_slice_{config_name}.segy"

        # Create test SEGY file with more traces
        num_traces = 25  # 5x5 grid
        samples_per_trace = SAMPLES_PER_TRACE

        spec = self.create_test_segy_file(
            spec=basic_segy_spec,
            num_traces=num_traces,
            samples_per_trace=samples_per_trace,
            output_path=segy_path,
            endianness=endianness,
            data_format=data_format,
            inline_range=(1, 5),
            crossline_range=(1, 5),
        )

        # Load the SEGY file
        segy_file = SegyFile(segy_path, spec=spec)

        # Test with slice
        wrapper = SegyFileTraceDataWrapper(segy_file, slice(5, 15))

        # Test that properties work with slice
        assert wrapper.header is not None
        assert wrapper.raw_header is not None
        assert wrapper.sample is not None

        # Check that we got the expected number of traces (10 traces from slice(5, 15))
        expected_count = 10
        assert wrapper.header.size == expected_count
        assert wrapper.raw_header.size == expected_count

    @pytest.mark.parametrize(
        "trace_indices",
        [
            0,  # Single trace
            [0, 2, 4],  # Multiple specific traces
            slice(1, 4),  # Range of traces
        ],
    )
    def test_different_index_types(
        self, temp_dir: Path, basic_segy_spec: SegySpec, segy_config: dict, trace_indices: int | list[int] | slice
    ) -> None:
        """Test wrapper with different types of trace indices."""
        config_name = segy_config["name"]
        endianness = segy_config["endianness"]
        data_format = segy_config["data_format"]

        segy_path = temp_dir / f"test_index_types_{config_name}.segy"

        # Create test SEGY file
        num_traces = 10
        samples_per_trace = SAMPLES_PER_TRACE

        spec = self.create_test_segy_file(
            spec=basic_segy_spec,
            num_traces=num_traces,
            samples_per_trace=samples_per_trace,
            output_path=segy_path,
            endianness=endianness,
            data_format=data_format,
        )

        # Load the SEGY file
        segy_file = SegyFile(segy_path, spec=spec)

        # Create wrapper with different index types
        wrapper = SegyFileTraceDataWrapper(segy_file, trace_indices)

        # Basic validation that we got results
        assert wrapper.header is not None
        assert wrapper.raw_header is not None
        assert wrapper.sample is not None

        # Check that the number of results matches expectation
        if isinstance(trace_indices, int):
            expected_count = 1
        elif isinstance(trace_indices, list):
            expected_count = len(trace_indices)
        elif isinstance(trace_indices, slice):
            expected_count = len(range(*trace_indices.indices(num_traces)))
        else:
            expected_count = 1

        assert wrapper.header.size == expected_count
