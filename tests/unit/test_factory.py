"""Test module for MDIO creation."""

from datetime import datetime

from mdio import MDIOReader
from mdio.core.factory import create_empty_like


def test_create_empty_like(mock_reader: MDIOReader):
    """Test create_empty_like function to ensure it replicates an existing MDIO dataset."""
    # Define destination path for the new dataset
    dest_path = mock_reader.url + "_copy"

    # Call create_empty_like
    create_empty_like(
        source_path=mock_reader.url,
        dest_path=dest_path,
        overwrite=True,
    )

    source_reader = mock_reader
    dest_reader = MDIOReader(dest_path)
    assert source_reader.grid.dims == dest_reader.grid.dims
    assert source_reader.live_mask != dest_reader.grid.live_mask

    source_traces = source_reader._traces
    dest_traces = dest_reader._traces

    assert source_traces.dtype == dest_traces.dtype
    assert source_traces.shape == dest_traces.shape
    assert source_traces.chunks == dest_traces.chunks
    assert source_traces.compressor == dest_traces.compressor

    source_headers = source_reader._headers
    dest_headers = dest_reader._headers

    assert source_headers.dtype == dest_headers.dtype
    assert source_headers.shape == dest_headers.shape
    assert source_headers.chunks == dest_headers.chunks
    assert source_headers.compressor == dest_headers.compressor

    assert source_reader.text_header == dest_reader.text_header
    assert source_reader.binary_header == dest_reader.binary_header

    # Verify live_mask
    assert dest_reader.live_mask[:].sum() == 0

    # Verify attributes
    assert dest_reader.trace_count == 0
    for stat_value in dest_reader.stats.values():
        assert stat_value == 0

    # Verify creation time is recent
    source_time = datetime.fromisoformat(source_reader.root.attrs["created"])
    dest_time = datetime.fromisoformat(dest_reader.root.attrs["created"])
    assert (dest_time - source_time).total_seconds() > 0
