"""Test MDIO compatibility with older versions."""

from pathlib import Path

import numpy as np
import pytest
import zarr
from segy import SegyFile
from segy.factory import SegyFactory
from segy.standards import get_segy_standard

from mdio import mdio_to_segy
from mdio import segy_to_mdio


# Constants
MDIO_VERSIONS = ["0.7.4", "0.8.3"]
SEGY_REVISIONS = [0.0, 0.1, 1.0, 1.1]
INLINES = (10, 10, 11, 11)
CROSSLINES = (100, 101, 100, 101)
INDEX_BYTES = (189, 193)
API_VERSION_KEY = "api_version"
BINARY_HEADER_KEY = "binary_header"
CHUNKED_TRACE_HEADERS_KEY = "chunked_012_trace_headers"


def update_mdio_for_version_0_7_4(root):
    """Update MDIO metadata to mimic version 0.7.4."""
    # Update binary header revision keys
    bin_hdr = root.metadata.attrs[BINARY_HEADER_KEY]
    bin_hdr["SEGYRevision"] = bin_hdr.pop("segy_revision_major")
    bin_hdr["SEGYRevisionMinor"] = bin_hdr.pop("segy_revision_minor")
    root.metadata.attrs[BINARY_HEADER_KEY] = bin_hdr

    # Remove trace headers past field 232 (pre-0.8 schema)
    orig_hdr = root.metadata[CHUNKED_TRACE_HEADERS_KEY]
    new_dtype = np.dtype(orig_hdr.dtype.descr[:-1])
    new_hdr = zarr.zeros_like(orig_hdr, dtype=new_dtype)
    root.metadata.create_dataset(
        CHUNKED_TRACE_HEADERS_KEY,
        data=new_hdr,
        overwrite=True,
    )
    zarr.consolidate_metadata(root.store)


@pytest.mark.parametrize("mdio_version", MDIO_VERSIONS)
@pytest.mark.parametrize("segy_revision", SEGY_REVISIONS)
def test_revision_encode_decode(
    mdio_version: str, segy_revision: float, tmp_path: Path
) -> None:
    """Test binary header major/minor revision roundtrip.

    After introducting TGSAI/segy, we changed the header names. Now we use
    aliasing and MDIO has a dummy schema. The handling is slightly different
    for SEG-Y revision major/minor numbers. Testing to ensure they're
    (de)serialized correctly.
    """
    rev1_spec = get_segy_standard(1.0)
    segy_filename = tmp_path / "segy_input.sgy"
    mdio_output_filename = tmp_path / "output.mdio"
    roundtrip_sgy_filename = tmp_path / "roundtrip_output.sgy"

    # Make a rev1 segy
    factory = SegyFactory(rev1_spec, sample_interval=1000, samples_per_trace=5)

    # We will replace the values in revision fields with these
    minor, major = np.modf(segy_revision)
    major, minor = int(major), int(minor * 10)
    revision_code = (major << 8) | minor

    # Make fake tiny 3D dataset
    txt_buffer = factory.create_textual_header()

    header = factory.create_trace_header_template(len(INLINES))
    data = factory.create_trace_sample_template(len(INLINES))
    header["inline"] = INLINES
    header["crossline"] = CROSSLINES
    data[:] = np.arange(len(INLINES))[:, None]
    trace_buffer = factory.create_traces(header, data)

    # Update revision during bin hdr creation
    bin_hdr_buffer = factory.create_binary_header(
        update={"segy_revision": revision_code}
    )
    with open(segy_filename, mode="wb") as fp:
        fp.write(txt_buffer)
        fp.write(bin_hdr_buffer)
        fp.write(trace_buffer)

    # Convert SEG-Y to MDIO
    segy_to_mdio(str(segy_filename), str(mdio_output_filename), index_bytes=INDEX_BYTES)

    # Modify MDIO for specific versions
    root = zarr.open_group(mdio_output_filename, mode="r+")
    root.attrs[API_VERSION_KEY] = mdio_version
    if mdio_version == "0.7.4":
        update_mdio_for_version_0_7_4(root)

    # Convert MDIO back to SEG-Y
    mdio_to_segy(str(mdio_output_filename), str(roundtrip_sgy_filename))

    # Assert binary headers and revisions match
    orig = SegyFile(segy_filename, spec=rev1_spec)
    rt = SegyFile(roundtrip_sgy_filename, spec=rev1_spec)
    assert orig.binary_header["segy_revision_major"] == major
    assert orig.binary_header["segy_revision_minor"] == minor
    assert orig.binary_header == rt.binary_header
