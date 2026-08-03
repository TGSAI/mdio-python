# Continuous Receiver Gathers

This guide covers the `NodalContinuousReceiverGathers3D` template for importing continuously recorded receiver data — land nodes and OBN — into MDIO.

## What makes this data different

Continuously recording receivers are powered on, left to record for weeks or months, and recovered later. There are no shots to index on. A delivered SEG-Y trace is one fixed-length segment of a single receiver's recording, and the only header that reliably tells segments apart is a 64-bit microsecond epoch timestamp.

This gives the data two distinct time axes: absolute wall-clock time, which selects a segment, and the sample axis within a segment.

## Template Overview

The `NodalContinuousReceiverGathers3D` template organizes data with the following dimensions:

Dimensions are ordered `receiver_line`, `receiver`, `component`, `segment_index`, `time`, following the field hierarchy: line, then station, then the sensors at that station, then position in time. Keeping `component` next to `segment_index` means all components of one station are adjacent on disk, so a multi-component read of a station stays local.

| Dimension       | Description                                                                                   |
| --------------- | --------------------------------------------------------------------------------------------- |
| `receiver_line` | Receiver line identifier                                                                      |
| `receiver`      | Receiver (station) identifier                                                                 |
| `component`     | Sensor component (e.g., 1=X, 2=Y, 3=Z, 4=Hydrophone)                                          |
| `segment_index` | Calculated dense position in time (see [Calculated segment_index](#calculated-segment_index)) |
| `time`          | Sample axis within one segment                                                                |

### Coordinates

- **Logical coordinates**: `epoch` — exact segment start time in microseconds, spanning the full spatial key `(receiver_line, receiver, component, segment_index)`
- **Physical coordinates**: `group_coord_x`, `group_coord_y` — indexed by `(receiver_line, receiver)`, since a receiver does not move during its deployment

## Calculated segment_index

`segment_index` is a calculated dimension, declared the same way `ObnReceiverGathers3D` declares `shot_index`. It is filled at ingest by the `CalculateSegmentIndex` grid override, which ranks each trace's `epoch` among the sorted unique epochs of its own `(receiver_line, receiver, component)` group. The override is mandatory: without it, ingestion fails because nothing else can supply the dimension.

### Why epoch cannot be the dimension

Receivers are powered on by hand, so their segment boundaries generally do not line up. No two receivers in a delivery are expected to share an epoch value, which means an `epoch` axis has one entry per trace and the grid becomes `receivers × traces` — a sparsity ratio equal to the receiver count. A 200-receiver deliverable holding four million traces produces a billion-cell grid, which will not build.

Ranking epochs per receiver collapses this to a dense grid:

```
Before (absolute epoch, µs, staggered starts):
  Receiver 101: 1556582074780000, 1556582104780000, 1556582134780000, ...
  Receiver 102: 1556582081780000, 1556582111780000, 1556582141780000, ...

After (dense segment_index):
  Receiver 101: 0, 1, 2, ...
  Receiver 102: 0, 1, 2, ...
```

### Why the ranking uses the header, not file order

A receiver's segments are commonly split across several field records that are not written in time order, so a counter that numbers traces by their position in the file will place segments at the wrong index. Ranking sorts on the `epoch` value itself, so the resulting axis is ordered by recording time regardless of how the vendor laid out the file.

### Epoch values are preserved exactly

`segment_index` alone cannot recover wall-clock time, because segment 0 of one receiver may start seconds after segment 0 of another. Ingestion therefore never modifies, rounds, or snaps the epoch: it is written as an `int64` microsecond coordinate spanning every spatial dimension, so a live segment's true recording time remains recoverable and receivers stay comparable across files.

### Components are ranked independently

Each component is its own ranking group. If two components of one receiver recorded different segment counts, a given `segment_index` may hold different wall-clock times for each of them. Because `epoch` spans `component`, every cell still reports the true start time of the segment stored there — read `epoch` rather than assuming the axis is aligned across components.

### Ragged receivers and the epoch fill value

Receivers in one deliverable often record slightly different segment counts, so the `segment_index` axis is sized to the longest receiver and shorter ones leave empty cells. Those cells have `trace_mask == False`. For `epoch`, empty cells hold the int64 fill sentinel (`np.iinfo(np.int64).max`). Always gate wall-clock queries on `trace_mask` (or an equivalent live-cell test); a one-sided `epoch >= T` filter alone will include dead cells.

## Chunking

The default chunk shape is `(1, 1, 1, 180, 15001)` over `(receiver_line, receiver, component, segment_index, time)`: about 10.3 MiB of float32, a little over the 8 MiB shot-template budget. Compression shrinks the stored size further anyway.

- **One receiver and one component per chunk.** A chunk never mixes recordings from different stations or sensors.
- **180 segments along `segment_index`.** 90 minutes of 30-second data. Wall-clock window reads on one receiver are consecutive on this axis, so packing segments cuts object-store request count without mixing receivers.
- **15,001 samples along `time`.** Matches a 30-second slice at 2 ms sample interval, so a typical segment fits one time chunk with no leftover samples. Longer sub-sampled records at or under that length also fit.

To change chunking for a different access pattern, set it on the template:

```python
template = get_template("NodalContinuousReceiverGathers3D")
template.full_chunk_shape = (1, 1, 1, 64, 8192)
```

## Special Behaviors

### Component Synthesis

When the SEG-Y spec does not include a `component` field, MDIO automatically synthesizes it with value `1` for all traces, so single-component deliverables use the same template unmodified.

## Usage

### Basic Import

```python
from segy.schema import HeaderField
from segy.standards import get_segy_standard

from mdio import GridOverrides
from mdio import segy_to_mdio
from mdio.builder.template_registry import get_template

crg_headers = [
    HeaderField(name="orig_field_record_num", byte=9, format="int32"),
    HeaderField(name="channel", byte=13, format="int32"),
    HeaderField(name="coordinate_scalar", byte=71, format="int16"),
    HeaderField(name="group_coord_x", byte=81, format="int32"),
    HeaderField(name="group_coord_y", byte=85, format="int32"),
    HeaderField(name="receiver_line", byte=137, format="int16"),
    HeaderField(name="receiver", byte=139, format="int16"),
    HeaderField(name="epoch", byte=189, format="int64"),
    HeaderField(name="component", byte=237, format="int16"),
]

crg_spec = get_segy_standard(1.0).customize(trace_header_fields=crg_headers)

segy_to_mdio(
    input_path="continuous_receiver_gathers.sgy",
    output_path="continuous_receiver_gathers.mdio",
    segy_spec=crg_spec,
    mdio_template=get_template("NodalContinuousReceiverGathers3D"),
    grid_overrides=GridOverrides(calculate_segment_index=True),
    overwrite=True,
)
```

```{note}
Some vendors document the epoch as two 32-bit words rather than one 64-bit field. Declaring a single `int64` field at the first byte reads the same value, so no header arithmetic is needed during ingestion.
```

### Exploring the Data

```python
import numpy as np

from mdio import open_mdio

ds = open_mdio("continuous_receiver_gathers.mdio")

print(ds.sizes)
# {'receiver_line': 1, 'receiver': 232, 'component': 1, 'segment_index': 27108, 'time': 15001}

# Absolute start time of every segment
print(ds["epoch"].dims)  # ('receiver_line', 'receiver', 'component', 'segment_index')

# Find the live segments of one receiver covering a wall-clock window
receiver = ds.sel(component=1, receiver_line=10, receiver=101)
live = receiver["trace_mask"]
window = live & (receiver["epoch"] >= 1556582074780000) & (receiver["epoch"] < 1556582674780000)
receiver["amplitude"].isel(segment_index=np.flatnonzero(window)).plot()
```

## Required Header Fields

| Field               | Required | Notes                                 |
| ------------------- | -------- | ------------------------------------- |
| `receiver_line`     | Yes      |                                       |
| `receiver`          | Yes      |                                       |
| `epoch`             | Yes      | 64-bit microsecond segment start time |
| `component`         | No       | Synthesized with value 1 if missing   |
| `coordinate_scalar` | Yes      |                                       |
| `group_coord_x`     | Yes      |                                       |
| `group_coord_y`     | Yes      |                                       |

## See Also

- [Grid Overrides](grid_overrides.md) - All available grid overrides
- [OBN Data Import](obn_data_import.md) - Shot-indexed OBN receiver gathers
- [Template Registry](../template_registry.md)
