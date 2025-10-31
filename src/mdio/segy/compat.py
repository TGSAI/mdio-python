"""Generate SEG-Y spec MDIO backward compatibility.

We were limited to fixed field names and byte locations due to using the segyio library. Since
MDIO 0.8.0 we have a more powerful SEG-Y parser and it gives more flexibility. To support older
files, we need to open them with the old SEG-Y spec. This is where we define it.
"""

from __future__ import annotations

import logging
from importlib import metadata

from packaging import version
from segy.alias.segyio import SEGYIO_BIN_FIELD_MAP
from segy.alias.segyio import SEGYIO_TRACE_FIELD_MAP
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import ScalarType
from segy.schema import SegySpec
from segy.schema import TextHeaderSpec
from segy.schema import TraceDataSpec
from segy.schema import TraceSpec
from segy.standards.fields import binary

from mdio.exceptions import InvalidMDIOError

MDIO_VERSION = metadata.version("multidimio")


logger = logging.getLogger(__name__)


def get_binary_fields() -> list[HeaderField]:
    """Generate binary header fields from equinor/segyio fields."""
    revision_field = binary.Rev1.SEGY_REVISION.model
    mdio_v0_bin_fields = []

    # Replace min/max (rev2-ish) with rev1 like parsing. Ignore minor one, and add the
    # revision as 4-byte.
    for alias, field in SEGYIO_BIN_FIELD_MAP.items():
        if alias == "SEGYRevision":
            mdio_v0_bin_fields.append(revision_field)
        elif alias != "SEGYRevisionMinor":
            mdio_v0_bin_fields.append(field.model)
    return mdio_v0_bin_fields


def get_trace_fields(version_str: str) -> list[HeaderField]:
    """Generate trace header fields.

    This part allows us to configure custom rules for different MDIO versions.

    For instance, since MDIO 0.8.0 we also save the unassigned parts of the trace header (after
    byte 233 / offset 232). To be able to ingest/export new MDIO files and also support exporting
    older MDIO files, we conditionally add the new field based on MDIO version specified above.

    Current rules:
    * mdio<=0.7.4 use the segyio mappings directly.
    * mdio>=0.8.0 adds an extra field to the end to fill the last 8 bytes

    Args:
        version_str: MDIO version to generate the trace fields for.

    Returns:
        List of header fields for specified MDIO version trace header encoding.
    """
    trace_fields = [field.model for field in SEGYIO_TRACE_FIELD_MAP.values()]
    version_obj = version.parse(version_str)
    if version_obj > version.parse("0.7.4"):
        trace_fields.append(HeaderField(name="unassigned", byte=233, format="int64"))
    return trace_fields


def mdio_segy_spec(version_str: str | None = None) -> SegySpec:
    """Get a SEG-Y encoding spec for MDIO based on version."""
    version_str = MDIO_VERSION if version_str is None else version_str

    binary_fields = get_binary_fields()
    trace_fields = get_trace_fields(version_str)

    return SegySpec(
        segy_standard=None,
        text_header=TextHeaderSpec(),
        binary_header=HeaderSpec(fields=binary_fields, item_size=400, offset=3200),
        trace=TraceSpec(
            header=HeaderSpec(fields=trace_fields, item_size=240),
            data=TraceDataSpec(format=ScalarType.IBM32),  # placeholder
        ),
    )


def encode_segy_revision(binary_header: dict) -> dict:
    """Encode revision code to binary header.

    Return the correctly Rev1-like encoded revision code, ready to write to SEG-Y.

    Args:
        binary_header: Dictionary representing the SEG-Y binary header. Contains keys for major
            and minor revision numbers.

    Returns:
        The updated binary header with the encoded revision.

    Raises:
        InvalidMDIOError: Raised when binary header in MDIO is broken.
    """
    major_key, minor_key = "segy_revision_major", "segy_revision_minor"

    try:
        major = binary_header.pop(major_key)
        minor = binary_header.pop(minor_key)
    except KeyError:
        msg = "Missing revision keys from binary header."
        logger.error(msg)
        raise InvalidMDIOError(msg) from KeyError

    code = (major << 8) | minor
    code_hex = f"0x{code:04x}"
    binary_header["segy_revision"] = code
    logger.info("Encoded revision %s.%s to code=%s ~ %s", major, minor, code, code_hex)
    return binary_header
