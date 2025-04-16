"""Test convenience functions in user facing API."""

import numpy as np
import pytest

from mdio import MDIOReader
from mdio import MDIOWriter
from mdio.api.convenience import copy_mdio
from mdio.api.convenience import rechunk_batch


def test_copy_without_data(mock_reader: MDIOReader):
    """Test MDIO copy with data excluding the data copy operation."""
    # Define destination path for the new dataset
    dest_path = mock_reader.url + "_copy"

    copy_mdio(
        source_path=mock_reader.url,
        target_path=dest_path,
        overwrite=True,
    )

    actual_reader = MDIOReader(dest_path)
    assert actual_reader.grid.dims == mock_reader.grid.dims

    # Expected mismatches
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        np.testing.assert_allclose(actual_reader._traces, mock_reader._traces)
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        np.testing.assert_array_equal(actual_reader._headers, mock_reader._headers)
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        np.testing.assert_array_equal(actual_reader.live_mask, mock_reader.live_mask)


@pytest.mark.dependency
def test_copy_with_data(mock_reader: MDIOReader):
    """Test MDIO copy with data included in the copy operation."""
    dest_path = mock_reader.url + "_copy2"

    copy_mdio(
        source_path=mock_reader.url,
        target_path=dest_path,
        overwrite=True,
        copy_headers=True,
        copy_traces=True,
    )

    actual_reader = MDIOReader(dest_path)
    assert actual_reader.grid.dims == mock_reader.grid.dims

    np.testing.assert_allclose(actual_reader._traces, mock_reader._traces)
    np.testing.assert_array_equal(actual_reader._headers, mock_reader._headers)
    np.testing.assert_array_equal(actual_reader.live_mask, mock_reader.live_mask)


@pytest.mark.dependency(depends=["test_copy_with_data"])
def test_rechunk(mock_reader: MDIOReader):
    """Test rechunking functionality."""
    dest_path = mock_reader.url + "_copy2"

    writer = MDIOWriter(dest_path)

    # Capture the original data and chunk sizes
    expected_traces = writer._traces[:]
    expected_headers = writer._headers[:]
    original_chunks = writer.chunks

    expected_chunks = (8, 8, 8)

    # Perform rechunking with a new suffix.
    rechunk_batch(writer, [expected_chunks], ["new_ap"], overwrite=True)

    # After rechunk, we need to reinitialize the reader to access the new chunks
    reader_new_ap = MDIOReader(dest_path, access_pattern="new_ap")

    # Get the rechunked data using the accessor's methods
    actual_traces = reader_new_ap._traces[:]
    actual_headers = reader_new_ap._headers[:]
    actual_chunks = reader_new_ap.chunks  # New chunk sizes

    # Validate that the underlying data has not changed.
    np.testing.assert_array_equal(actual_traces, expected_traces)
    np.testing.assert_array_equal(actual_headers, expected_headers)

    # Validate that the new chunk sizes match what we specified
    assert actual_chunks == expected_chunks
    assert actual_chunks != original_chunks
