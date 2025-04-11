"""Integration test for MDIO rechunking.

This test creates a fake 3D SEG‑Y file with a 3×4 grid (3 inlines and 4 crosslines)
with 100 samples per trace. Each trace header stores its inline and crossline numbers.
It then converts the SEG‑Y file to MDIO, reads the original data arrays, performs a
rechunk operation via the convenience API, and finally validates that the data in the
new rechunked arrays exactly matches the original MDIO data.
"""

import struct

import numpy as np
import pytest

from mdio.api import convenience
from mdio.api.accessor import MDIOAccessor
from mdio.converters import segy_to_mdio


def create_fake_segy_3d(file_path):
    """Create a fake 3D SEG-Y file with 3 inlines and 4 crosslines and 100 samples per trace.

    Each trace header includes inline and crossline numbers, stored in big-endian format
    at positions corresponding to the SEG-Y standard (bytes 189 and 193).
    """
    num_inlines = 3
    num_crosslines = 4
    samples_per_trace = 100

    with open(file_path, "wb") as f:
        # Write textual header (3200 bytes).
        f.write(b" " * 3200)
        # Create a binary header of 400 bytes using a mutable bytearray.
        bin_header = bytearray(400)
        # For SEG‑Y revision 0, the sample interval is stored at bytes 17–18 (0-indexed: 16:18).
        # Set the sample interval to 1000 microseconds.
        bin_header[16:18] = struct.pack(">H", 1000)
        # The number of samples per trace is stored at bytes 21–22 (0-indexed: 20:22).
        # Set the number of samples per trace to 100.
        bin_header[20:22] = struct.pack(">H", 100)
        # Optionally, set the data sample format code at bytes 25–26
        # (0-indexed: 24:26) to 5 (IEEE floating point).
        bin_header[24:26] = struct.pack(">H", 5)
        # Set bytes 96-99 to 0 so that explicit endianness code is 0
        # and the SEG-Y library will fall back to the legacy method.
        bin_header[96:100] = b"\x00" * 4
        f.write(bin_header)
        for inline in range(1, num_inlines + 1):
            for crossline in range(1, num_crosslines + 1):
                # Create a 240-byte trace header.
                header = bytearray(240)
                # SEG‑Y standard:
                # - Inline number is stored at bytes 189-192.
                # - Crossline number is stored at bytes 193-196.
                # - Python indexing is 0-based.
                header[188:192] = struct.pack(">i", inline)
                header[192:196] = struct.pack(">i", crossline)
                f.write(header)
                # Create trace sample data.
                # For each IL/XL pair, we increment the base value by 1, and for each trace
                # (i.e. each sample) we increment by 0.002.
                trace_samples = np.arange(
                    samples_per_trace, dtype=np.float32
                ) * 0.002 + (inline * 10 + crossline + 1)
                # Convert samples to big-endian IEEE float32 before writing
                trace_samples_be = trace_samples.astype(">f4")
                f.write(trace_samples_be.tobytes())


@pytest.fixture
def segy_file(tmp_path):
    """Create a fake 3D SEG-Y file with 3 inlines and 4 crosslines and 100 samples per trace."""
    segy_path = tmp_path / "fake3d.sgy"
    create_fake_segy_3d(segy_path)
    return segy_path


@pytest.fixture
def mdio_path(tmp_path):
    """Create a temporary MDIO file."""
    return tmp_path / "test.mdio"


def test_rechunk_integration(segy_file, mdio_path):
    """Basic rechunking test.

    1. Convert a fake 3D SEG-Y file to an MDIO file.
    2. Capture the original data arrays from the resulting MDIO file.
    3. Perform a rechunk operation via the convenience API.
    4. Validate that the rechunked arrays have the same underlying data as the original,
       ensuring that data integrity remains undamaged.
    5. Validate that the chunking in zarr metadata matches the expected chunk sizes.
    """
    # Convert the fake SEG-Y file to MDIO.
    # For conversion, we choose inline and crossline header values from bytes 189 and 193.
    segy_to_mdio(
        segy_path=str(segy_file),
        mdio_path_or_buffer=str(mdio_path),
        index_bytes=(189, 193),
        index_names=("inline", "crossline"),
        chunksize=(2, 2, 100),
        overwrite=True,
    )

    # Create an MDIOReader for the newly created MDIO file.
    reader = MDIOAccessor(
        str(mdio_path),
        mode="r+",
        access_pattern="012",
        storage_options=None,
        return_metadata=False,
        new_chunks=None,
        backend="zarr",
        memory_cache_size=0,
        disk_cache=False,
    )

    # Capture the original data and chunk sizes
    original_traces = reader._traces[
        :
    ]  # Main data array (3D: inline, crossline, samples).
    original_headers = reader._headers[:]  # Header array.
    original_chunks = reader._traces.chunks  # Original chunk sizes

    # Verify original chunking
    assert original_chunks == (2, 2, 100), "Original chunk sizes do not match expected"

    # Choose a new chunk size different from the original.
    # Here we change the chunking of the inline dimension.
    new_chunk = (3, 4, 50)

    # Perform rechunking with a new suffix.
    convenience.rechunk(reader, new_chunk, "sample", overwrite=True)

    # After rechunk, we need to reinitialize the reader to access the new chunks
    rechunked_reader = MDIOAccessor(
        str(mdio_path),
        mode="r+",
        access_pattern="sample",
        storage_options=None,
        return_metadata=False,
        new_chunks=None,
        backend="zarr",
        memory_cache_size=0,
        disk_cache=False,
    )

    # Get the rechunked data using the accessor's methods
    rechunked_data = rechunked_reader._traces[:]
    rechunked_headers = rechunked_reader._headers[:]
    rechunked_chunks = rechunked_reader._traces.chunks  # New chunk sizes

    # Validate that the underlying data has not changed.
    np.testing.assert_array_equal(original_traces, rechunked_data)
    np.testing.assert_array_equal(original_headers, rechunked_headers)

    # Validate that the new chunk sizes match what we specified
    assert rechunked_chunks == new_chunk, "Rechunked sizes do not match expected"
    assert original_chunks != rechunked_chunks, "Chunk sizes should have changed"
