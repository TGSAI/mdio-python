# Constructing a v1 Dataset with the MDIODatasetBuilder

In this tutorial, we'll walk through how to use the `MDIODatasetBuilder` class to programmatically construct an MDIO v1 dataset. The builder enforces a specific build order to ensure a valid dataset:

1. Add dimensions via `add_dimension()`
2. (Optional) Add coordinates via `add_coordinate()`
3. Add variables via `add_variable()`
4. Call `build()` to finalize the dataset.

## Importing the Builder

```python
from mdio.core.v1.builder import MDIODatasetBuilder, write_mdio_metadata
from mdio.schemas.dtype import ScalarType, StructuredType
from mdio.schemas.compressors import Blosc, ZFP
```

## Creating the Builder

First, create a builder instance with a name and optional global attributes:

```python
builder = MDIODatasetBuilder(
    name="example_dataset",
    attributes={
        "description": "An example MDIO v1 dataset",
        "creator": "Your Name",
    },
)
```

## Adding Dimensions

Dimensions define the axes of your dataset. You must add at least one dimension before adding coordinates or variables:

```python
builder = (
    builder
    .add_dimension(name="inline", size=256, long_name="Inline Number")
    .add_dimension(name="crossline", size=512, long_name="Crossline Number")
    .add_dimension(name="depth", size=384, long_name="Depth Sample")
)
```

## Adding Coordinates (Optional)

Coordinates map grid indices to real-world positions (e.g., UTM coordinates on the inline–crossline plane):

```python
builder = (
    builder
    .add_coordinate(
        name="cdp_x",
        dimensions=["inline", "crossline"],
        long_name="CDP X (UTM Easting)",
        data_type=ScalarType.FLOAT64,
        metadata={"unitsV1": {"length": "m"}},
    )
    .add_coordinate(
        name="cdp_y",
        dimensions=["inline", "crossline"],
        long_name="CDP Y (UTM Northing)",
        data_type=ScalarType.FLOAT64,
        metadata={"unitsV1": {"length": "m"}},
    )
)
```

If you omit `name`, the builder auto-generates names like `coord_0`. If you omit `dimensions`, it uses all defined dimensions.

## Adding Variables

Add one or more seismic data variables (e.g., post-stack amplitude volumes). Variables can have compressors, statistics, and more:

```python
builder = builder.add_variable(
    name="stack_amplitude",
    dimensions=["inline", "crossline", "depth"],
    data_type=ScalarType.FLOAT32,
    compressor=Blosc(algorithm="zstd", level=3),
    coordinates=["inline", "crossline", "cdp_x", "cdp_y"],
    metadata={
        "chunkGrid": {"name": "regular", "configuration": {"chunkShape": [64, 64, 64]}}
    },
)
```

For structured dtypes, use `StructuredType`:

```python
from mdio.schemas.dtype import StructuredType, ScalarType

structured_dtype = StructuredType(
    fields=[
        {"name": "flag", "format": ScalarType.INT8},
        {"name": "value", "format": ScalarType.FLOAT32},
    ]
)

builder = builder.add_variable(
    name="metadata",
    dimensions=["x", "y"],
    data_type=structured_dtype,
)
```

## Building the Dataset

After adding all components, call:

```python
dataset = builder.build()
```

This returns a `Dataset` object conforming to the MDIO v1 schema.

## Writing Metadata and Writing Data

The `.build()` method returns an in-memory Pydantic `Dataset` model (MDIO v1 schema). To serialize this model to disk, use the following approaches:

- **Metadata only** (no array values written):

  ```python
  # Write metadata structure only (no data arrays)
  mds = write_mdio_metadata(
      dataset,
      store="path/to/output.mdio"
  )
  ```

  This writes only the metadata to the `.mdio` store and returns an `mdio.Dataset` (an xarray.Dataset subclass) with placeholder arrays.

- **Write actual data** (array values):

  After writing metadata, call `to_mdio()` on the returned `mdio.Dataset` with `compute=True` to write the actual data arrays:

  ```python
  # Write data arrays into the existing store
  mds.to_mdio(
      store="path/to/output.mdio",
      mode="a",
      compute=True,
  )
  ```

  Alternatively, skip `write_mdio_metadata()` and write both metadata and data in one call by invoking `to_mdio()` directly on the `mdio.Dataset` produced by `_construct_mdio_dataset`, if you have it available.
