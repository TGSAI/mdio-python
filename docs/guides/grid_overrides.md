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

## HasDuplicates

Inserts a `trace` dimension to disambiguate traces that share the same index tuple, by
appending a dense per-tuple counter (in trace order) as the `trace` axis before the vertical
axis. Used for continuous receiver gathers, where many segments share the same
`(component, receiver_line, receiver)` index.

**Supported Templates:** any template (commonly `ObnContinuousReceiverGathers3D`)

**Parameters:**

| Parameter     | Default | Description                                                               |
| ------------- | ------- | ------------------------------------------------------------------------- |
| `chunksize`   | `1`     | Chunk size for the inserted `trace` dimension.                            |
| `trace_dtype` | -       | Integer dtype for the `trace` counter and the coordinate it is stored as. |

Both parameters are optional and backwards compatible: omitting them reproduces the legacy
behavior (chunk size `1`, an `int16` counter stored as an `int32` coordinate). An `int16`
counter tops out at 32,767 traces per index tuple and raises `OverflowError` beyond that, so
set `trace_dtype` for larger gathers. `trace_dtype` applies to `NonBinned` as well, since it
inserts the same `trace` axis.

**Usage:**

```python
from mdio import GridOverrides

# ~64 MiB chunks for a whole-trace (15001-sample) continuous receiver gather.
GridOverrides(has_duplicates=True, chunksize=1024, trace_dtype="uint32")
```

```{note}
See [CRG Data Import](crg_data_import.md) for a complete guide on importing continuous
receiver gathers.
```

## Special Behaviors

Some templates have special behaviors that are applied automatically during import, independent of grid overrides.

### Component Synthesis (OBN)

When using the `ObnReceiverGathers3D` template, if the SEG-Y specification does not include a `component` field, MDIO automatically synthesizes it with value `1` for all traces. This allows single-component data (e.g., hydrophone-only) to use the same template without modification.

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

- [OBN Data Import](obn_data_import.md) - Complete guide for OBN data
- [Template Registry](../template_registry.md) - Available templates
- [Tutorials](../tutorials/index.md) - Hands-on guides
