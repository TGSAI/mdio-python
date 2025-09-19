"""Test harness for disaster recovery wrapper.

This module tests the _disaster_recovery_wrapper.py functionality by creating
test SEGY files with different configurations and validating that the raw headers
from get_header_raw_and_transformed match the bytes on disk.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from segy import SegyFile
from segy.factory import SegyFactory
from segy.schema import Endianness
from segy.schema import HeaderField
from segy.schema import SegySpec
from segy.standards import get_segy_standard

from mdio.segy._disaster_recovery_wrapper import get_header_raw_and_transformed

if TYPE_CHECKING:
    from numpy.typing import NDArray

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

    def extract_header_bytes_from_file(
        self, segy_path: Path, trace_index: int, byte_start: int, byte_length: int
    ) -> NDArray:
        """Extract specific bytes from a trace header in the SEGY file."""
        with segy_path.open("rb") as f:
            # Skip text header (3200 bytes) + binary header (400 bytes)
            header_offset = 3600

            # Each trace: 240 byte header + samples
            trace_size = 240 + SAMPLES_PER_TRACE * 4  # samples * 4 bytes each
            trace_offset = header_offset + trace_index * trace_size

            f.seek(trace_offset + byte_start - 1)  # SEGY is 1-based
            header_bytes = f.read(byte_length)

            return np.frombuffer(header_bytes, dtype=np.uint8)

    def test_header_validation_configurations(
        self, temp_dir: Path, basic_segy_spec: SegySpec, segy_config: dict
    ) -> None:
        """Test header validation with different SEGY configurations."""
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

        # Test a few traces
        test_indices = [0, 3, 7]

        for trace_idx in test_indices:
            # Get raw and transformed headers
            raw_headers, transformed_headers, traces = get_header_raw_and_transformed(
                segy_file=segy_file, indices=trace_idx, do_reverse_transforms=True
            )

            # Extract bytes from disk for inline (bytes 189-192) and crossline (bytes 193-196)
            inline_bytes_disk = self.extract_header_bytes_from_file(segy_path, trace_idx, 189, 4)
            crossline_bytes_disk = self.extract_header_bytes_from_file(segy_path, trace_idx, 193, 4)

            # Convert raw headers to bytes for comparison
            if raw_headers is not None:
                # Extract from raw headers
                # Note: We need to extract bytes directly from the structured array to preserve endianness
                # Getting a scalar and calling .tobytes() loses endianness information
                if raw_headers.ndim == 0:
                    # Single trace case
                    raw_data_bytes = raw_headers.tobytes()
                    inline_offset = raw_headers.dtype.fields["inline"][1]
                    crossline_offset = raw_headers.dtype.fields["crossline"][1]
                    inline_size = raw_headers.dtype.fields["inline"][0].itemsize
                    crossline_size = raw_headers.dtype.fields["crossline"][0].itemsize

                    raw_inline_bytes = np.frombuffer(
                        raw_data_bytes[inline_offset : inline_offset + inline_size], dtype=np.uint8
                    )
                    raw_crossline_bytes = np.frombuffer(
                        raw_data_bytes[crossline_offset : crossline_offset + crossline_size], dtype=np.uint8
                    )
                else:
                    # Multiple traces case - this test uses single trace index, so extract that trace
                    raw_data_bytes = raw_headers[0:1].tobytes()  # Extract first trace
                    inline_offset = raw_headers.dtype.fields["inline"][1]
                    crossline_offset = raw_headers.dtype.fields["crossline"][1]
                    inline_size = raw_headers.dtype.fields["inline"][0].itemsize
                    crossline_size = raw_headers.dtype.fields["crossline"][0].itemsize

                    raw_inline_bytes = np.frombuffer(
                        raw_data_bytes[inline_offset : inline_offset + inline_size], dtype=np.uint8
                    )
                    raw_crossline_bytes = np.frombuffer(
                        raw_data_bytes[crossline_offset : crossline_offset + crossline_size], dtype=np.uint8
                    )

                print(f"Transformed headers: {transformed_headers.tobytes()}")
                print(f"Raw headers: {raw_headers.tobytes()}")
                print(f"Inline bytes disk: {inline_bytes_disk.tobytes()}")
                print(f"Crossline bytes disk: {crossline_bytes_disk.tobytes()}")

                # Compare bytes
                assert np.array_equal(raw_inline_bytes, inline_bytes_disk), (
                    f"Inline bytes mismatch for trace {trace_idx} in {config_name}"
                )
                assert np.array_equal(raw_crossline_bytes, crossline_bytes_disk), (
                    f"Crossline bytes mismatch for trace {trace_idx} in {config_name}"
                )

    def test_header_validation_no_transforms(
        self, temp_dir: Path, basic_segy_spec: SegySpec, segy_config: dict
    ) -> None:
        """Test header validation when transforms are disabled."""
        config_name = segy_config["name"]
        endianness = segy_config["endianness"]
        data_format = segy_config["data_format"]

        segy_path = temp_dir / f"test_no_transforms_{config_name}.segy"

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

        # Get headers without reverse transforms
        raw_headers, transformed_headers, traces = get_header_raw_and_transformed(
            segy_file=segy_file,
            indices=slice(None),  # All traces
            do_reverse_transforms=False,
        )

        # When transforms are disabled, raw_headers should be None
        assert raw_headers is None

        # Transformed headers should still be available
        assert transformed_headers is not None
        assert transformed_headers.size == num_traces

    def test_multiple_traces_validation(self, temp_dir: Path, basic_segy_spec: SegySpec, segy_config: dict) -> None:
        """Test validation with multiple traces at once."""
        config_name = segy_config["name"]
        endianness = segy_config["endianness"]
        data_format = segy_config["data_format"]

        print(f"Config name: {config_name}")
        print(f"Endianness: {endianness}")
        print(f"Data format: {data_format}")

        segy_path = temp_dir / f"test_multiple_traces_{config_name}.segy"

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

        # Get all traces
        raw_headers, transformed_headers, traces = get_header_raw_and_transformed(
            segy_file=segy_file,
            indices=slice(None),  # All traces
            do_reverse_transforms=True,
        )

        first = True

        # Validate each trace
        for trace_idx in range(num_traces):
            # Extract bytes from disk
            inline_bytes_disk = self.extract_header_bytes_from_file(segy_path, trace_idx, 189, 4)
            crossline_bytes_disk = self.extract_header_bytes_from_file(segy_path, trace_idx, 193, 4)

            if first:
                print(raw_headers.dtype)
                print(raw_headers.shape)
                first = False

            # Extract from raw headers
            # Note: We need to extract bytes directly from the structured array to preserve endianness
            # Getting a scalar and calling .tobytes() loses endianness information
            if raw_headers.ndim == 0:
                # Single trace case
                raw_data_bytes = raw_headers.tobytes()
                inline_offset = raw_headers.dtype.fields["inline"][1]
                crossline_offset = raw_headers.dtype.fields["crossline"][1]
                inline_size = raw_headers.dtype.fields["inline"][0].itemsize
                crossline_size = raw_headers.dtype.fields["crossline"][0].itemsize

                raw_inline_bytes = np.frombuffer(
                    raw_data_bytes[inline_offset : inline_offset + inline_size], dtype=np.uint8
                )
                raw_crossline_bytes = np.frombuffer(
                    raw_data_bytes[crossline_offset : crossline_offset + crossline_size], dtype=np.uint8
                )
            else:
                # Multiple traces case
                raw_data_bytes = raw_headers[trace_idx : trace_idx + 1].tobytes()
                inline_offset = raw_headers.dtype.fields["inline"][1]
                crossline_offset = raw_headers.dtype.fields["crossline"][1]
                inline_size = raw_headers.dtype.fields["inline"][0].itemsize
                crossline_size = raw_headers.dtype.fields["crossline"][0].itemsize

                raw_inline_bytes = np.frombuffer(
                    raw_data_bytes[inline_offset : inline_offset + inline_size], dtype=np.uint8
                )
                raw_crossline_bytes = np.frombuffer(
                    raw_data_bytes[crossline_offset : crossline_offset + crossline_size], dtype=np.uint8
                )

            print(f"Raw inline bytes: {raw_inline_bytes.tobytes()}")
            print(f"Inline bytes disk: {inline_bytes_disk.tobytes()}")
            print(f"Raw crossline bytes: {raw_crossline_bytes.tobytes()}")
            print(f"Crossline bytes disk: {crossline_bytes_disk.tobytes()}")

            # Compare
            assert np.array_equal(raw_inline_bytes, inline_bytes_disk), (
                f"Inline bytes mismatch for trace {trace_idx} in {config_name}"
            )
            assert np.array_equal(raw_crossline_bytes, crossline_bytes_disk), (
                f"Crossline bytes mismatch for trace {trace_idx} in {config_name}"
            )

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
        """Test with different types of trace indices."""
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

        # Get headers with different index types
        raw_headers, transformed_headers, traces = get_header_raw_and_transformed(
            segy_file=segy_file, indices=trace_indices, do_reverse_transforms=True
        )

        # Basic validation that we got results
        assert raw_headers is not None
        assert transformed_headers is not None
        assert traces is not None

        # Check that the number of results matches expectation
        if isinstance(trace_indices, int):
            expected_count = 1
        elif isinstance(trace_indices, list):
            expected_count = len(trace_indices)
        elif isinstance(trace_indices, slice):
            expected_count = len(range(*trace_indices.indices(num_traces)))
        else:
            expected_count = 1

        assert transformed_headers.size == expected_count
