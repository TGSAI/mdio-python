```{eval-rst}
:tocdepth: 3
```

```{currentModule} mdio.schema.chunk_grid

```

# Chunk Grid Models

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

The variables in MDIO data model can represent different types of chunk grids.
These grids are essential for managing multi-dimensional data arrays efficiently.
In this breakdown, we will explore four distinct data models within the MDIO schema,
each serving a specific purpose in data handling and organization.

MDIO implements data models following the guidelines of the Zarr v3 spec and ZEPs:

- [Zarr core specification (version 3)](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html)
- [ZEP 1 — Zarr specification version 3](https://zarr.dev/zeps/accepted/ZEP0001.html)
- [ZEP 3 — Variable chunking](https://zarr.dev/zeps/draft/ZEP0003.html)

## Regular Grid

The regular grid models are designed to represent a rectangular and regularly
paced chunk grid.

```{eval-rst}
.. autosummary::
   RegularChunkGrid
   RegularChunkShape
```

For 1D array with `size = 31`{l=python}, we can divide it into 5 equally sized
chunks. Note that the last chunk will be truncated to match the size of the array.

`{ "name": "regular", "configuration": { "chunkShape": [7] } }`{l=json}

Using the above schema resulting array chunks will look like this:

```bash
 ←─ 7 ─→ ←─ 7 ─→ ←─ 7 ─→ ←─ 7 ─→  ↔ 3
┌───────┬───────┬───────┬───────┬───┐
└───────┴───────┴───────┴───────┴───┘
```

For 2D array with shape `rows, cols = (7, 17)`{l=python}, we can divide it into 9
equally sized chunks.

`{ "name": "regular", "configuration": { "chunkShape": [3, 7] } }`{l=json}

Using the above schema, the resulting 2D array chunks will look like below.
Note that the rows and columns are conceptual and visually not to scale.

```bash
 ←─ 7 ─→ ←─ 7 ─→  ↔ 3
┌───────┬───────┬───┐
│       ╎       ╎   │  ↑
│       ╎       ╎   │  3
│       ╎       ╎   │  ↓
├╶╶╶╶╶╶╶┼╶╶╶╶╶╶╶┼╶╶╶┤
│       ╎       ╎   │  ↑
│       ╎       ╎   │  3
│       ╎       ╎   │  ↓
├╶╶╶╶╶╶╶┼╶╶╶╶╶╶╶┼╶╶╶┤
│       ╎       ╎   │  ↕ 1
└───────┴───────┴───┘
```

## Rectilinear Grid

The [RectilinearChunkGrid](RectilinearChunkGrid) model extends
the concept of chunk grids to accommodate rectangular and irregularly spaced chunks.
This model is useful in data structures where non-uniform chunk sizes are necessary.
[RectilinearChunkShape](RectilinearChunkShape) specifies the chunk sizes for each
dimension as a list allowing for irregular intervals.

```{eval-rst}
.. autosummary::
   RectilinearChunkGrid
   RectilinearChunkShape
```

:::{note}
It's important to ensure that the sum of the irregular spacings specified
in the `chunkShape` matches the size of the respective array dimension.
:::

For 1D array with `size = 39`{l=python}, we can divide it into 5 irregular sized
chunks.

`{ "name": "rectilinear", "configuration": { "chunkShape": [[10, 7, 5, 7, 10]] } }`{l=json}

Using the above schema resulting array chunks will look like this:

```bash
 ←── 10 ──→ ←─ 7 ─→ ← 5 → ←─ 7 ─→ ←── 10 ──→
┌──────────┬───────┬─────┬───────┬──────────┐
└──────────┴───────┴─────┴───────┴──────────┘
```

For 2D array with shape `rows, cols = (7, 25)`{l=python}, we can divide it into 12
rectilinear (rectangular bur irregular) chunks. Note that the rows and columns are
conceptual and visually not to scale.

`{ "name": "rectilinear", "configuration": { "chunkShape": [[3, 1, 3], [10, 5, 7, 3]] } }`{l=json}

```bash
 ←── 10 ──→ ← 5 → ←─ 7 ─→  ↔ 3
┌──────────┬─────┬───────┬───┐
│          ╎     ╎       ╎   │  ↑
│          ╎     ╎       ╎   │  3
│          ╎     ╎       ╎   │  ↓
├╶╶╶╶╶╶╶╶╶╶┼╶╶╶╶╶┼╶╶╶╶╶╶╶┼╶╶╶┤
│          ╎     ╎       ╎   │  ↕ 1
├╶╶╶╶╶╶╶╶╶╶┼╶╶╶╶╶┼╶╶╶╶╶╶╶┼╶╶╶┤
│          ╎     ╎       ╎   │  ↑
│          ╎     ╎       ╎   │  3
│          ╎     ╎       ╎   │  ↓
└──────────┴─────┴───────┴───┘
```

## Model Reference

:::{dropdown} RegularChunkGrid
:animate: fade-in-slide-down

```{eval-rst}
.. autopydantic_model:: RegularChunkGrid

----------

.. autopydantic_model:: RegularChunkShape
```

:::
:::{dropdown} RectilinearChunkGrid
:animate: fade-in-slide-down

```{eval-rst}
.. autopydantic_model:: RectilinearChunkGrid

----------

.. autopydantic_model:: RectilinearChunkShape
```

:::
