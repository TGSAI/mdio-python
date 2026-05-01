# Grid Overrides

```{warning}
🚧👷🏻 We are actively working on updating the documentation and adding missing features to v1 release. Please check back later for more updates!
```

Grid overrides are transformations applied during SEG-Y import that modify how trace headers are interpreted and indexed. They handle complex acquisition geometries that cannot be represented by simple header-to-dimension mappings.

## Overview

When importing SEG-Y data, MDIO maps trace header fields to dataset dimensions. However, real-world seismic data often has complexities that require additional processing. Grid overrides address these issues by transforming header values before indexing.

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
segy_to_mdio(
    input_path="obn_data.sgy",
    output_path="obn_data.mdio",
    segy_spec=obn_spec,
    mdio_template=get_template("ObnReceiverGathers3D"),
    grid_overrides={"CalculateShotIndex": True},
)
```

```{note}
See [OBN Data Import](obn_data_import.md) for a complete guide on importing OBN data.
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
