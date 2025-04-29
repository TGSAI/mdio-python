```{eval-rst}
:tocdepth: 3
```

```{currentModule} mdio.schemas.dtype

```

# Data Types

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

## Scalar Type

Scalar types are used to represent numbers and boolean values in MDIO arrays.

```{eval-rst}
.. autosummary::
   :nosignatures:

   ScalarType
```

These numbers can be integers (whole numbers without a decimal
point, like 1, -15, 204), floating-point numbers (numbers with a fractional part,
like 3.14, -0.001, 2.71828) in various 16-64 bit formats like `float32` etc.

It is important to choose the right type for the content of the data for type safety,
memory efficiency, performance, and accuracy of the numbers represented. Most scientific
datasets are `float16`, `float32`, or `float64` values. However, there are many good
use cases for integer and complex values as well.

The [`ScalarType`](#ScalarType)s MDIO supports can be viewed below with the tabs.

:::::{tab-set}

::::{tab-item} Boolean
:::{table}
:widths: auto
:align: center

| Data Type | Options         | Example Value |
| --------- | --------------- | ------------- |
| `bool`    | `False`, `True` | `True`        |

:::
::::

::::{tab-item} Integers
:::{table}
:widths: auto
:align: center

| Data Type | Range                                                       | Example Value |
| --------- | ----------------------------------------------------------- | ------------- |
| `int8`    | `-128` to `127`                                             | `45`          |
| `int16`   | `-32,768` to `32,767`                                       | `1,234`       |
| `int32`   | `-2,147,483,648` to `2,147,483,647`                         | `2,024`       |
| `int64`   | `-9,223,372,036,854,775,808` to `9,223,372,036,854,775,807` | `987,654,321` |

:::
::::

::::{tab-item} Unsigned Integers
:::{table}
:widths: auto
:align: center

| Data Type | Range                               | Example Value   |
| --------- | ----------------------------------- | --------------- |
| `uint8`   | `0` to `255`                        | `200`           |
| `uint16`  | `0` to `65,535`                     | `50,000`        |
| `uint32`  | `0` to `4,294,967,295`              | `3,000,000`     |
| `uint64`  | `0` to `18,446,744,073,709,551,615` | `5,000,000,000` |

:::
::::

::::{tab-item} Floating Point
:::{table}
:widths: auto
:align: center

| Data Type | Range                                                   | Example Value        |
| --------- | ------------------------------------------------------- | -------------------- |
| `float16` | `-65,504` to `65,504`                                   | `10.10`              |
| `float32` | `-3.4028235e+38` to `3.4028235e+38`                     | `0.1234567`          |
| `float64` | `-1.7976931348623157e+308` to `1.7976931348623157e+308` | `3.1415926535897932` |

:::

**Precision**

- `float16`: 2 decimal places
- `float32`: 7 decimal places
- `float32`: 16 decimal places

::::

::::{tab-item} Complex Numbers
:::{table}
:widths: auto
:align: center

| Data Type    | Range                                                   | Example Value      |
| ------------ | ------------------------------------------------------- | ------------------ |
| `complex64`  | `-3.4028235e+38` to `3.4028235e+38`                     | `3.14+2.71j`       |
| `complex128` | `-1.7976931348623157e+308` to `1.7976931348623157e+308` | `2.71828+3.14159j` |

:::
Ranges are for both real and imaginary parts.
::::

:::::

## Structured Type

Structured data type organizes and stores data in a fixed arrangement, allowing memory
efficient access and manipulation.

```{eval-rst}
.. autosummary::
   :nosignatures:

   StructuredType
   StructuredField
```

Structured data types are an essential component in handling complex data structures,
particularly in specialized domains like seismic data processing for subsurface
imaging applications. These data types allow for the organization of heterogeneous
data into a single, structured format.

They are designed to be memory-efficient, which is vital for handling large seismic
datasets. Structured data types are adaptable, allowing for the addition or
modification of fields.

A [`StructuredType`](#StructuredType) consists of [`StructuredField`](#StructuredField)s.
Fields can be different [numeric types](#numeric-types), and each represent a specific
attribute of the seismic data, like coordinate, line numbers, and time stamps.

Each [`StructuredField`](#StructuredField) must specify a `name` and a data format
(`format`).

All the structured fields will be packed and there will be no gaps between them.

## Examples

The table below illustrate [ScalarType](#ScalarType) ranges and shows an example each
type.

Variable `foo` with type `float32`.

```json
{
  "name": "foo",
  "dataType": "float32",
  "dimensions": ["x", "y"]
}
```

Variable `bar` with type `uint8`.

```json
{
  "name": "bar",
  "dataType": "uint8",
  "dimensions": ["x", "y"]
}
```

Below are a couple examples of [StructuredType](#StructuredType) with varying lengths.

We can specify a variable named `headers` that holds a 32-byte struct with
four `int32` values.

```json
{
  "name": "headers",
  "dataType": {
    "fields": [
      { "name": "cdp-x", "format": "int32" },
      { "name": "cdp-y", "format": "int32" },
      { "name": "inline", "format": "int32" },
      { "name": "crossline", "format": "int32" }
    ]
  },
  "dimensions": ["inline", "crossline"]
}
```

This will yield an in-memory or on-disk struct that looks like this (for each element):

```bash
 ←─ 4 ─→ ←─ 4 ─→ ←─ 4 ─→ ←─ 4 ─→  = 16-bytes
┌───────┬───────┬───────┬───────┐
│ int32 ╎ int32 ╎ int32 ╎ int32 │ ⋯ (next sample)
└───────┴───────┴───────┴───────┘
  └→ cdp-x └→ cdp-y └→ inline └→crossline
```

The below example shows mixing different data types.

```json
{
  "name": "headers",
  "dataType": {
    "fields": [
      { "name": "cdp", "format": "uint32" },
      { "name": "offset", "format": "int16" },
      { "name": "cdp-x", "format": "float64" },
      { "name": "cdp-y", "format": "float64" }
    ]
  },
  "dimensions": ["inline", "crossline"]
}
```

This will yield an in-memory or on-disk struct that looks like this (for each element):

```bash
 ←── 4 ──→ ← 2 → ←─── 8 ───→ ←─── 8 ───→  = 24-bytes
┌─────────┬─────┬───────────┬───────────┐
│  int32  ╎int16╎  float64  ╎  float64  │ ⋯ (next sample)
└─────────┴─────┴───────────┴───────────┘
    └→ cdp  └→ offset └→ cdp-x    └→ cdp-y
```

## Model Reference

:::{dropdown} Scalar Types
:animate: fade-in-slide-down

```{eval-rst}
.. autoclass:: ScalarType()
    :members:
    :undoc-members:
    :member-order: bysource
```

:::

:::{dropdown} Structured Type
:animate: fade-in-slide-down

```{eval-rst}
.. autopydantic_model:: StructuredType

----------

.. autopydantic_model:: StructuredField
```

:::
