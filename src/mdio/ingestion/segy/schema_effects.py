"""SEG-Y grid-override schema reshapes.

These :class:`~mdio.ingestion.schema.models.SchemaEffect` implementations reshape a
:class:`~mdio.ingestion.schema.models.ResolvedSchema` for trace-producing grid overrides.
They live in the SEG-Y package because their vocabulary (``trace``, ``NonBinned``,
``HasDuplicates``) is SEG-Y specific; the
:class:`~mdio.ingestion.segy.index_strategies.IndexStrategyRegistry` selects the right one.
"""

from __future__ import annotations

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.templates.types import CoordinateSpec
from mdio.ingestion.schema.models import DimensionSpec
from mdio.ingestion.schema.models import ResolvedSchema
from mdio.ingestion.schema.models import SchemaEffect

_TRACE_DIM = "trace"


class InsertTraceDimEffect(SchemaEffect):
    """Insert a calculated trace dimension before the vertical axis.

    Args:
        chunksize: Chunk size for the trace dimension.
    """

    def __init__(self, chunksize: int = 1) -> None:
        self.chunksize = chunksize

    def apply(self, schema: ResolvedSchema) -> ResolvedSchema:
        """Insert the trace dimension and its chunk before the vertical dimension."""
        spatial_dims = [dim for dim in schema.dimensions if dim.is_spatial]
        spatial_chunks = [
            chunk for dim, chunk in zip(schema.dimensions, schema.chunk_shape, strict=True) if dim.is_spatial
        ]

        trace_dim = DimensionSpec(name=_TRACE_DIM, is_spatial=True, is_calculated=True)
        new_dimensions = [*spatial_dims, trace_dim]
        new_chunk_shape = [*spatial_chunks, self.chunksize]

        for dim, chunk in zip(schema.dimensions, schema.chunk_shape, strict=True):
            if not dim.is_spatial:
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk)

        return schema.model_copy(update={"dimensions": new_dimensions, "chunk_shape": tuple(new_chunk_shape)})


class CollapseToTraceEffect(SchemaEffect):
    """Collapse spatial dimensions into a single trace dimension.

    Args:
        chunksize: Chunk size for the trace dimension.
        collapse_dims: Names of spatial dimensions to collapse. If None, collapses
            all spatial dimensions except the first.
    """

    def __init__(self, chunksize: int | None, collapse_dims: tuple[str, ...] | None = None) -> None:
        self.chunksize = chunksize
        self.collapse_dims = collapse_dims

    def _resolve_collapse_dims(self, schema: ResolvedSchema) -> tuple[str, ...]:
        """Resolve the spatial dimensions to collapse."""
        if self.collapse_dims is not None:
            return self.collapse_dims
        spatial = schema.spatial_dimensions()
        return tuple(dim.name for dim in spatial[1:]) if len(spatial) > 1 else ()

    def apply(self, schema: ResolvedSchema) -> ResolvedSchema:
        """Collapse spatial dimensions into trace and rewrite coordinates."""
        collapse = self._resolve_collapse_dims(schema)
        collapse_set = set(collapse)

        spatial_dims = [
            (dim, chunk)
            for dim, chunk in zip(schema.dimensions, schema.chunk_shape, strict=True)
            if dim.is_spatial and dim.name not in collapse_set
        ]

        replaced_count = sum(1 for dim in schema.dimensions if dim.name in collapse_set)

        new_dimensions = [dim for dim, _ in spatial_dims]
        new_chunk_shape = [chunk for _, chunk in spatial_dims]

        if replaced_count > 0:
            new_dimensions.append(DimensionSpec(name=_TRACE_DIM, is_spatial=True, is_calculated=True))
            new_chunk_shape.append(self.chunksize)

        for dim, chunk in zip(schema.dimensions, schema.chunk_shape, strict=True):
            if not dim.is_spatial:
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk)

        new_coordinates = self._rewrite_coordinates(schema, collapse, collapse_set, replaced_count)

        return schema.model_copy(
            update={
                "dimensions": new_dimensions,
                "coordinates": new_coordinates,
                "chunk_shape": tuple(new_chunk_shape),
            }
        )

    @staticmethod
    def _rewrite_coordinates(
        schema: ResolvedSchema,
        collapse: tuple[str, ...],
        collapse_set: set[str],
        replaced_count: int,
    ) -> list[CoordinateSpec]:
        """Rewrite coordinate dimensions and add collapsed dimensions as trace coordinates."""
        rewritten: list[CoordinateSpec] = []
        for coord in schema.coordinates:
            had_collapsed = any(name in collapse_set for name in coord.dimensions)
            dims = tuple(name for name in coord.dimensions if name not in collapse_set)
            if had_collapsed and replaced_count > 0:
                dims = (*dims, _TRACE_DIM)
            rewritten.append(coord.model_copy(update={"dimensions": dims}))

        existing = {coord.name for coord in rewritten}
        dtype_by_dim = {dim.name: dim.dtype for dim in schema.dimensions}
        for name in collapse:
            if name in existing:
                continue
            rewritten.append(
                CoordinateSpec(
                    name=name,
                    dimensions=(_TRACE_DIM,),
                    dtype=dtype_by_dim.get(name, ScalarType.INT32),
                )
            )
        return rewritten
