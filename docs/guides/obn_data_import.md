# OBN Data Import

This guide covers the `ObnReceiverGathers3D` template for importing Ocean Bottom Node (OBN) seismic data into MDIO.

## Template Overview

The `ObnReceiverGathers3D` template organizes data with the following dimensions:

| Dimension      | Description                                                                                |
| -------------- | ------------------------------------------------------------------------------------------ |
| `component`    | Sensor component (e.g., 1=X, 2=Y, 3=Z, 4=Hydrophone)                                       |
| `receiver`     | Ocean bottom node receiver ID                                                              |
| `shot_line`    | Shot line identifier                                                                       |
| `gun`          | Gun identifier for multi-gun sources                                                       |
| `shot_index`   | Calculated dense index for shots (see [Required Grid Overrides](#required-grid-overrides)) |
| `time`/`depth` | Vertical sample axis                                                                       |

### Coordinates

- **Logical coordinates**: `shot_point` (original values), `orig_field_record_num`
- **Physical coordinates**: `group_coord_x`, `group_coord_y`, `source_coord_x`, `source_coord_y`

```{note}
The `shot_index` dimension is calculated (0 to N-1) from `shot_point` values during ingestion. Original `shot_point` values are preserved as a coordinate indexed by `(shot_line, gun, shot_index)`.
```

## Required Grid Overrides

### CalculateShotIndex (Required)

The `CalculateShotIndex` grid override is **required** for the `ObnReceiverGathers3D` template. It calculates the `shot_index` dimension from `shot_point` values. Without this override, the import will fail with an error:

> Required computed fields ['shot_index'] for template ObnReceiverGathers3D not found after grid overrides.

This override handles multi-gun acquisition where shot points are interleaved across guns, calculating a dense `shot_index` from sparse `shot_point` values:

```
Before (interleaved shot_point):
  Gun 1: 1, 3, 5, 7, ...
  Gun 2: 2, 4, 6, 8, ...

After (dense shot_index):
  Gun 1: 0, 1, 2, 3, ...
  Gun 2: 0, 1, 2, 3, ...
```

For `ObnReceiverGathers3D`, the override uses `shot_line` as the line field and requires `shot_line`, `gun`, and `shot_point` headers.

## Special Behaviors

### Component Synthesis

When the SEG-Y spec does not include a `component` field, MDIO automatically synthesizes it with value `1` for all traces. This allows single-component data (e.g., hydrophone-only) to use the same template without modification.

```{note}
A warning is logged when component is synthesized:
> SEG-Y headers do not contain 'component' field required by template 'ObnReceiverGathers3D'.
> Synthesizing 'component' dimension with constant value 1 for all traces.
```

## Usage

### Basic Import

```python
from segy.schema import HeaderField
from segy.standards import get_segy_standard

from mdio import segy_to_mdio
from mdio.builder.template_registry import get_template

# Define SEG-Y header mapping
obn_headers = [
    HeaderField(name="orig_field_record_num", byte=9, format="int32"),
    HeaderField(name="receiver", byte=13, format="int32"),
    HeaderField(name="shot_point", byte=17, format="int32"),
    HeaderField(name="shot_line", byte=133, format="int16"),
    HeaderField(name="gun", byte=171, format="int16"),
    HeaderField(name="component", byte=189, format="int16"),
    HeaderField(name="coordinate_scalar", byte=71, format="int16"),
    HeaderField(name="source_coord_x", byte=73, format="int32"),
    HeaderField(name="source_coord_y", byte=77, format="int32"),
    HeaderField(name="group_coord_x", byte=81, format="int32"),
    HeaderField(name="group_coord_y", byte=85, format="int32"),
]

obn_spec = get_segy_standard(1.0).customize(trace_header_fields=obn_headers)

segy_to_mdio(
    input_path="obn_data.sgy",
    output_path="obn_data.mdio",
    segy_spec=obn_spec,
    mdio_template=get_template("ObnReceiverGathers3D"),
    grid_overrides={"CalculateShotIndex": True},
    overwrite=True,
)
```

### Single-Component Data

For data without a `component` header field, simply omit it from the spec:

```python
# Same as above, but without the component field
obn_headers = [
    HeaderField(name="orig_field_record_num", byte=9, format="int32"),
    HeaderField(name="receiver", byte=13, format="int32"),
    HeaderField(name="shot_point", byte=17, format="int32"),
    HeaderField(name="shot_line", byte=133, format="int16"),
    HeaderField(name="gun", byte=171, format="int16"),
    # component omitted - will be synthesized
    HeaderField(name="coordinate_scalar", byte=71, format="int16"),
    HeaderField(name="source_coord_x", byte=73, format="int32"),
    HeaderField(name="source_coord_y", byte=77, format="int32"),
    HeaderField(name="group_coord_x", byte=81, format="int32"),
    HeaderField(name="group_coord_y", byte=85, format="int32"),
]
```

### Exploring the Data

```python
from mdio import open_mdio

ds = open_mdio("obn_data.mdio")

# View dimensions
print(ds.sizes)
# {'component': 4, 'receiver': 100, 'shot_line': 10, 'gun': 2, 'shot_index': 500, 'time': 2001}

# Access original shot_point values (preserved as coordinate)
print(ds["shot_point"].dims)   # ('shot_line', 'gun', 'shot_index')

# Select a receiver gather
receiver_gather = ds.sel(receiver=150, component=4)
receiver_gather["amplitude"].plot()
```

## Required Header Fields

| Field                   | Required | Notes                               |
| ----------------------- | -------- | ----------------------------------- |
| `receiver`              | Yes      |                                     |
| `shot_line`             | Yes      |                                     |
| `gun`                   | Yes      |                                     |
| `shot_point`            | Yes      |                                     |
| `component`             | No       | Synthesized with value 1 if missing |
| `coordinate_scalar`     | Yes      |                                     |
| `source_coord_x`        | Yes      |                                     |
| `source_coord_y`        | Yes      |                                     |
| `group_coord_x`         | Yes      |                                     |
| `group_coord_y`         | Yes      |                                     |
| `orig_field_record_num` | Yes      |                                     |

## See Also

- [Grid Overrides](grid_overrides.md) - All available grid overrides
- [Template Registry](../template_registry.md)
- [Quickstart Tutorial](../tutorials/quickstart.ipynb)
