# Grid Overrides

```{warning}
🚧👷🏻 We are actively working on updating the documentation and adding missing features to v1 release. Please check back later for more updates!
```

Grid overrides are transformations applied during SEG-Y import that modify how trace headers are interpreted and indexed. They handle complex acquisition geometries that cannot be represented by simple header-to-dimension mappings.

## Overview

When importing SEG-Y data, MDIO maps trace header fields to dataset dimensions. However, real-world seismic data often has complexities that require additional processing. Grid overrides address these issues by transforming header values before indexing.

## Configuring grid overrides

Grid overrides are passed to {func}`mdio.segy_to_mdio` via the `grid_overrides` argument as an
{class}`mdio.GridOverrides` instance:

```python
from mdio import GridOverrides
from mdio import segy_to_mdio

segy_to_mdio(
    ...,
    grid_overrides=GridOverrides(calculate_shot_index=True),
)
```

Both modern `snake_case` field names and the legacy `CamelCase` aliases are accepted, so
`GridOverrides(CalculateShotIndex=True)` is equivalent to the example above. Unknown keys
are rejected at construction with a `pydantic.ValidationError`.

```{deprecated} 1.2
Passing `grid_overrides` as a `dict` still works but logs a deprecation warning and will be
removed in a future release. Switch to `mdio.GridOverrides`.
```

## CalculateShotIndex

Calculates a dense `shot_index` dimension from sparse or interleaved `shot_point` values. Required for the `ObnReceiverGathers3D` template.

**Supported Templates:** `ObnReceiverGathers3D`

**Required Headers:** `shot_line`, `gun`, `shot_point`

**How it works:**

In multi-gun OBN acquisition, shot points are often interleaved across guns:

```
Before (interleaved shot_point):
  Gun 1: 1, 3, 5, 7, ...
  Gun 2: 2, 4, 6, 8, ...

After (dense shot_index):
  Gun 1: 0, 1, 2, 3, ...
  Gun 2: 0, 1, 2, 3, ...
```

The override detects the geometry type and only applies the transformation when shot points are actually interleaved (Type B geometry). For non-interleaved data (Type A), shot points are used directly.

**Usage:**

```python
from mdio import GridOverrides
from mdio import segy_to_mdio

segy_to_mdio(
    input_path="obn_data.sgy",
    output_path="obn_data.mdio",
    segy_spec=obn_spec,
    mdio_template=get_template("ObnReceiverGathers3D"),
    grid_overrides=GridOverrides(calculate_shot_index=True),
)
```

```{note}
See [OBN Data Import](obn_data_import.md) for a complete guide on importing OBN data.
```

### CalculateSegmentIndex

**Purpose:** Build the dense `segment_index` axis for continuously recording receivers by ranking each trace's `epoch` header.

**Supported Templates:** `NodalContinuousReceiverGathers3D`

**Required Headers:** `epoch`, plus the template's receiver dimensions (`receiver_line`, `receiver`, `component`)

Each trace's index is the position of its `epoch` among the sorted unique epochs of its own `(receiver_line, receiver, component)` group, so the axis is ordered by recording time however the vendor laid out the file. Epoch values are never modified; they are preserved as an `int64` microsecond coordinate.

```
Before (absolute epoch, µs, staggered starts):
  Receiver 101: 1556582074780000, 1556582104780000, 1556582134780000, ...
  Receiver 102: 1556582081780000, 1556582111780000, 1556582141780000, ...

After (dense segment_index):
  Receiver 101: 0, 1, 2, ...
  Receiver 102: 0, 1, 2, ...
```

**Usage:**

```python
from mdio import GridOverrides
from mdio import segy_to_mdio

segy_to_mdio(
    input_path="continuous_receivers.sgy",
    output_path="continuous_receivers.mdio",
    segy_spec=crg_spec,
    mdio_template=get_template("NodalContinuousReceiverGathers3D"),
    grid_overrides=GridOverrides(calculate_segment_index=True),
)
```

The override is mandatory for this template: nothing else can supply `segment_index`, so ingestion fails without it. It cannot be combined with `HasDuplicates` or `NonBinned`, whose trace counters would group on the freshly computed index.

**Chunking:** unlike `HasDuplicates`, this override takes no chunking arguments. `segment_index` is a dimension declared by the template, so the template owns its chunk shape and the override only fills in the index values. To change chunking, set it on the template instead:

```python
template = get_template("NodalContinuousReceiverGathers3D")
template.full_chunk_shape = (1, 1, 1, 64, 8192)  # receiver_line, receiver, component, segment_index, time
```

```{note}
See [Continuous Receiver Gathers](continuous_receiver_gathers.md) for the full guide, including
chunking rationale, the epoch coordinate, and ragged-grid fill semantics.
```

## Special Behaviors

Some templates have special behaviors that are applied automatically during import, independent of grid overrides.

### Component Synthesis

The `ObnReceiverGathers3D` and `NodalContinuousReceiverGathers3D` templates declare `component` in `synthesize_missing_dims`. If the SEG-Y specification does not include a `component` field, MDIO automatically synthesizes it with value `1` for all traces. This allows single-component data (e.g., hydrophone-only) to use the same template without modification.

```{note}
A warning is logged when component is synthesized:

> SEG-Y headers do not contain 'component' field required by template 'ObnReceiverGathers3D'.
> Synthesizing 'component' dimension with constant value 1 for all traces.
```

## Error Handling

Grid overrides validate their requirements and raise specific exceptions:

| Exception                           | Cause                               |
| ----------------------------------- | ----------------------------------- |
| `GridOverrideUnknownError`          | Unknown override name passed        |
| `GridOverrideKeysError`             | Required header fields missing      |
| `GridOverrideMissingParameterError` | Required parameters not provided    |
| `GridOverrideIncompatibleError`     | Override incompatible with template |

**Example error message:**

```
GridOverrideKeysError: Grid override 'CalculateShotIndex' requires keys: {'shot_line', 'gun', 'shot_point'}
```

## See Also

- [Continuous Receiver Gathers](continuous_receiver_gathers.md) - Template-owned segment ranking
- [OBN Data Import](obn_data_import.md) - Complete guide for OBN data
- [Template Registry](../template_registry.md) - Available templates
- [Tutorials](../tutorials/index.md) - Hands-on guides
