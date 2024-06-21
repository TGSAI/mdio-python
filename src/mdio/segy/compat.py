"""Generate SEG-Y spec for legacy MDIO pre version 0.8.0.

We were limited to fixed field names and byte locations due to
using the segyio library. Now we have a more powerful SEG-Y parser
and it gives more flexibility. To support older files, we need to
open them with the old SEG-Y spec. This is where we define it.
"""

from segy.alias.segyio import SEGYIO_BIN_FIELD_MAP
from segy.alias.segyio import SEGYIO_TRACE_FIELD_MAP
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import ScalarType
from segy.schema import SegySpec
from segy.schema import TextHeaderSpec
from segy.schema import TraceDataSpec
from segy.schema import TraceSpec


binary_fields = [field.model for field in SEGYIO_BIN_FIELD_MAP.values()]
trace_fields = [field.model for field in SEGYIO_TRACE_FIELD_MAP.values()]
trace_fields.append(HeaderField(name="unassigned", byte=233, format="int64"))

mdio_segyio_spec = SegySpec(
    segy_standard=None,
    text_header=TextHeaderSpec(),  # default EBCDIC
    binary_header=HeaderSpec(fields=binary_fields, item_size=400, offset=3200),
    trace=TraceSpec(
        header=HeaderSpec(fields=trace_fields, item_size=240),
        data=TraceDataSpec(format=ScalarType.IBM32),  # placeholder
    ),
)
