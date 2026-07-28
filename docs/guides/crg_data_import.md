# CRG Data Import

This guide covers the `ObnContinuousReceiverGathers3D` template for importing OBN
**Continuous Receiver Gathers** (CRG, clock-time) seismic data into MDIO.

In OBN acquisition each ocean-bottom node records continuously. A continuous-receiver-gather
delivery bundles many receivers and splits each receiver's long recording into fixed-length
(e.g. 30 s) segments, one per SEG-Y trace. The template stores those segments along a dense
per-receiver axis while preserving each segment's absolute recording time in the headers.

## Template Overview

The `ObnContinuousReceiverGathers3D` template organizes data with the following dimensions:

| Dimension       | Description                                                                   |
| --------------- | ----------------------------------------------------------------------------- |
| `component`     | Sensor component (e.g., 1=X, 2=Y, 3=Z, 4=Hydrophone); synthesized when absent |
| `receiver_line` | Receiver line number                                                          |
| `receiver`      | Receiver station number                                                       |
| `trace`         | Dense per-receiver segment index, inserted at ingest (see below)              |
| `time`/`depth`  | Vertical sample axis (whole trace kept in one chunk)                          |

### Coordinates

- **Physical coordinates**: `group_coord_x`, `group_coord_y` (receiver X/Y), indexed by
  `(receiver_line, receiver)`.

```{note}
The `trace` dimension is **not** declared by the template. It is inserted during ingestion by
the [`HasDuplicates`](grid_overrides.md#hasduplicates) grid override, which appends a dense
per-`(component, receiver_line, receiver)` counter in trace order. This reuses MDIO's existing
duplicate-handling machinery instead of introducing a new calculated dimension.
```

## Required Grid Overrides

### HasDuplicates (Required)

Because many segments share the same `(component, receiver_line, receiver)` index tuple, the
`HasDuplicates` grid override is **required**. It inserts the `trace` segment dimension:

```python
from mdio import GridOverrides

# Whole 30 s trace is 15001 samples; a trace chunk of 1024 gives ~64 MiB chunks.
GridOverrides(has_duplicates=True, chunksize=1024)
```

- `chunksize` sizes the inserted `trace` dimension. It is optional and defaults to `1` (the
  legacy behavior), but a production CRG ingest should set it so chunks are a sensible size.
- `trace_dtype` sets the dtype of both the segment counter and the stored `trace` coordinate.
  Omitting it keeps the legacy `int16` counter stored as `int32`, which tops out at 32,767
  segments per receiver and raises `OverflowError` past that. Use `"uint32"` for
  full-campaign products, where a single receiver holds far more segments.

```{note}
Real recording time is **not** encoded into the segment index. The `trace` axis is a dense
positional index in acquisition order; each segment's absolute time is preserved per trace in
the SEG-Y headers (typically an `epoch` field), which rides through to any cut SEG-Y unchanged.
```

## Special Behaviors

### Component Synthesis

When the SEG-Y spec does not include a `component` field, MDIO synthesizes it with value `1`
for all traces, so one template ingests both single- and multi-component CRG data. This is
driven by the template's `synthesize_missing_dims` and handled by `ComponentSynthesisStrategy`.

```{note}
A warning is logged when component is synthesized:
> SEG-Y headers do not contain 'component' field required by template; synthesizing dimension
> with constant value 1 for all traces.
```

## Usage

### Single-Component Import

```python
from segy.schema import HeaderField
from segy.standards import get_segy_standard

from mdio import GridOverrides
from mdio import segy_to_mdio
from mdio.builder.template_registry import get_template

# CRG SEG-Y trace-header mapping (big-endian; bytes past 179 are vendor-defined).
crg_headers = [
    HeaderField(name="channel", byte=13, format="int32"),
    HeaderField(name="coordinate_scalar", byte=71, format="int16"),
    HeaderField(name="group_coord_x", byte=81, format="int32"),
    HeaderField(name="group_coord_y", byte=85, format="int32"),
    HeaderField(name="receiver_line", byte=137, format="int16"),
    HeaderField(name="receiver", byte=139, format="int16"),
    HeaderField(name="epoch", byte=189, format="int64"),  # per-trace recording time
]

crg_spec = get_segy_standard(1.0).customize(trace_header_fields=crg_headers)

segy_to_mdio(
    input_path="crg_data.sgy",
    output_path="crg_data.mdio",
    segy_spec=crg_spec,
    mdio_template=get_template("ObnContinuousReceiverGathers3D"),
    grid_overrides=GridOverrides(has_duplicates=True, chunksize=1024, trace_dtype="uint32"),
    overwrite=True,
)
```

### Exploring the Data

```python
from mdio import open_mdio

ds = open_mdio("crg_data.mdio")

# View dimensions
print(ds.sizes)
# {'component': 1, 'receiver_line': 1, 'receiver': 10, 'trace': 8172, 'time': 15001}

# Select a whole receiver gather
receiver_gather = ds.sel(receiver_line=4871, receiver=5908, component=1)
receiver_gather["amplitude"].plot()
```

## Required Header Fields

| Field               | Required | Notes                                       |
| ------------------- | -------- | ------------------------------------------- |
| `receiver_line`     | Yes      |                                             |
| `receiver`          | Yes      | Receiver station                            |
| `coordinate_scalar` | Yes      |                                             |
| `group_coord_x`     | Yes      | Receiver X                                  |
| `group_coord_y`     | Yes      | Receiver Y                                  |
| `component`         | No       | Synthesized with value 1 if missing         |
| `epoch`             | No       | Recommended; per-trace time kept in headers |

## See Also

- [Grid Overrides](grid_overrides.md) - All available grid overrides
- [OBN Data Import](obn_data_import.md) - Shot-indexed OBN receiver gathers
- [Template Registry](../template_registry.md)
