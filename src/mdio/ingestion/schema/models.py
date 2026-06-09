"""Immutable schema models that fully describe a dataset before any data is scanned."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.templates.types import CoordinateSpec  # noqa: TC001  (pydantic needs this at runtime)

if TYPE_CHECKING:
    from collections.abc import Iterable


class DimensionSpec(BaseModel):
    """Specification for a dataset dimension.

    Attributes:
        name: Name of the dimension.
        is_spatial: True if the dimension is spatial, False for the vertical/data-domain.
        is_calculated: True if coordinate values are produced by an index strategy.
        dtype: Coordinate data type.
    """

    name: str
    is_spatial: bool = True
    is_calculated: bool = False
    dtype: ScalarType = ScalarType.INT32


class ResolvedSchema(BaseModel):
    """Resolved schema for dataset ingestion.

    Attributes:
        name: Name of the dataset or template.
        dimensions: Specifications for the dimensions.
        coordinates: Specifications for the coordinates.
        chunk_shape: Chunk size for each dimension.
        metadata: Metadata attributes.
        default_variable_name: Name of the primary data variable.
    """

    name: str
    dimensions: list[DimensionSpec]
    coordinates: list[CoordinateSpec]
    chunk_shape: tuple[int, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)
    default_variable_name: str = "amplitude"

    def required_fields(self) -> set[str]:
        """Get names of fields required from the source to materialize this schema."""
        fields = {dim.name for dim in self.dimensions if dim.is_spatial and not dim.is_calculated}
        fields.update(coord.name for coord in self.coordinates)
        return fields

    def spatial_dimensions(self) -> list[DimensionSpec]:
        """Get spatial dimensions."""
        return [dim for dim in self.dimensions if dim.is_spatial]

    def missing_calculated_dimensions(self, produced_names: Iterable[str]) -> list[str]:
        """Get calculated spatial dimensions absent from actual produced dimensions.

        Args:
            produced_names: Names of dimensions actually produced.

        Returns:
            List of missing calculated spatial dimension names.
        """
        produced = set(produced_names)
        return [
            dim.name for dim in self.dimensions if dim.is_spatial and dim.is_calculated and dim.name not in produced
        ]


_TRACE_DIM = "trace"


class SchemaEffect(ABC):
    """Effect that reshapes a ResolvedSchema."""

    @abstractmethod
    def apply(self, schema: ResolvedSchema) -> ResolvedSchema:
        """Apply effect to schema and return a new ResolvedSchema."""


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
