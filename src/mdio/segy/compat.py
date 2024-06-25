"""Generate SEG-Y spec MDIO backward compatibility.

We were limited to fixed field names and byte locations due to using the segyio
library. Since MDIO 0.8.0 we have a more powerful SEG-Y parser and it gives more
flexibility. To support older files, we need to open them with the old SEG-Y spec.
This is where we define it.
"""

from __future__ import annotations

from importlib import metadata

from segy.alias.segyio import SEGYIO_BIN_FIELD_MAP
from segy.alias.segyio import SEGYIO_TRACE_FIELD_MAP
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import ScalarType
from segy.schema import SegySpec
from segy.schema import TextHeaderSpec
from segy.schema import TraceDataSpec
from segy.schema import TraceSpec


MDIO_VERSION = metadata.version("multidimio")


def get_binary_fields() -> list[HeaderField]:
    """Generate binary header fields from equinor/segyio fields."""
    return [field.model for field in SEGYIO_BIN_FIELD_MAP.values()]


def get_trace_fields(version: str) -> list[HeaderField]:
    """Generate trace header fields.

    This part allows us to configure custom rules for different MDIO versions.

    For instance, since MDIO 0.8.0 we also save the unassigned parts of the
    trace header (after byte 233 / offset 232). To be able to ingest/export
    new MDIO files and also support exporting older MDIO files, we conditionally
    add the new field based on MDIO version specified above.

    Current rules:
    * mdio<=0.7.4 use the segyio mappings directly.
    * mdio>=0.8.0 adds an extra field to the end to fill the last 8 bytes

    Args:
        version: MDIO version to generate the trace fields for.

    Returns:
        List of header fields for specified MDIO version trace header encoding.
    """
    trace_fields = [field.model for field in SEGYIO_TRACE_FIELD_MAP.values()]
    if version > "0.7.4":
        trace_fields.append(HeaderField(name="unassigned", byte=233, format="int64"))
    return trace_fields


def mdio_segy_spec(version: str | None = None) -> SegySpec:
    """Get a SEG-Y encoding spec for MDIO based on version."""
    version = MDIO_VERSION if version is None else version

    binary_fields = get_binary_fields()
    trace_fields = get_trace_fields(version)

    return SegySpec(
        segy_standard=None,
        text_header=TextHeaderSpec(),  # default EBCDIC
        binary_header=HeaderSpec(fields=binary_fields, item_size=400, offset=3200),
        trace=TraceSpec(
            header=HeaderSpec(fields=trace_fields, item_size=240),
            data=TraceDataSpec(format=ScalarType.IBM32),  # placeholder
        ),
    )
