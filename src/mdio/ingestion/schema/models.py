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
    """Specification for a dimension in the final dataset.

    Attributes:
        name: Dimension name (e.g. ``"inline"``, ``"shot_point"``, ``"trace"``, ``"time"``).
        is_spatial: Whether this is a spatial dimension. ``False`` only for the vertical
            (data-domain) dimension.
        is_calculated: Whether this dimension's coordinate values are produced by an index
            strategy at ingest time (e.g. ``shot_index`` from a template, or ``trace`` added
            by a grid override) rather than read directly. The pipeline uses this to give a
            clear error if a required strategy was not enabled.
        dtype: Data type for the dimension coordinate.
    """

    name: str
    is_spatial: bool = True
    is_calculated: bool = False
    dtype: ScalarType = ScalarType.INT32


class ResolvedSchema(BaseModel):
    """Final resolved schema for dataset ingestion.

    This represents the complete, resolved schema after applying template configuration
    and grid overrides. It contains everything needed to build the dataset without
    any further decision-making.

    Attributes:
        name: Name of the dataset/template
        dimensions: Ordered list of dimension specifications
        coordinates: List of coordinate specifications
        chunk_shape: Tuple of chunk sizes for each dimension
        metadata: Additional metadata attributes
        default_variable_name: Name of the main data variable
    """

    name: str
    dimensions: list[DimensionSpec]
    coordinates: list[CoordinateSpec]
    chunk_shape: tuple[int, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)
    default_variable_name: str = "amplitude"

    def required_fields(self) -> set[str]:
        """Names that must be readable from the source to materialize this schema."""
        fields = {dim.name for dim in self.dimensions if dim.is_spatial and not dim.is_calculated}
        fields.update(coord.name for coord in self.coordinates)
        return fields

    def spatial_dimensions(self) -> list[DimensionSpec]:
        """Get only spatial dimensions (excludes vertical/trace domain)."""
        return [dim for dim in self.dimensions if dim.is_spatial]

    def missing_calculated_dimensions(self, produced_names: Iterable[str]) -> list[str]:
        """Return calculated spatial dimensions that no index strategy produced.

        Calculated dimensions (e.g. ``shot_index``, or ``trace`` from a grid override) are
        not read directly from SEG-Y headers; an index strategy must materialize them. The
        pipeline uses this to fail clearly when the required strategy was not enabled.

        Args:
            produced_names: Names of dimensions actually produced after header analysis.

        Returns:
            Calculated spatial dimension names absent from ``produced_names``.
        """
        produced = set(produced_names)
        return [
            dim.name for dim in self.dimensions if dim.is_spatial and dim.is_calculated and dim.name not in produced
        ]


_TRACE_DIM = "trace"


class SchemaEffect(ABC):
    """How a trace-producing index strategy reshapes a :class:`ResolvedSchema`.

    A grid override changes both the trace headers (handled by an
    :class:`~mdio.ingestion.segy.index_strategies.IndexStrategy`) and the resolved schema
    layout (dimensions, chunk shape, coordinates). To keep those two views from drifting,
    the index-strategy registry is the single place that maps a ``GridOverrides`` to the
    matching :class:`SchemaEffect`; the :class:`~mdio.ingestion.schema.resolver.SchemaResolver`
    simply applies whatever effect it is handed.
    """

    @abstractmethod
    def apply(self, schema: ResolvedSchema) -> ResolvedSchema:
        """Return a new schema with this effect applied; ``schema`` is left unchanged."""


class InsertTraceDimEffect(SchemaEffect):
    """Insert a calculated ``trace`` dimension just before the vertical axis.

    Mirrors the ``HasDuplicates`` override: no spatial dimension is collapsed, a single
    extra ``trace`` axis disambiguates duplicate index tuples.

    Args:
        chunksize: Chunk size assigned to the inserted ``trace`` dimension.
    """

    def __init__(self, chunksize: int = 1) -> None:
        self.chunksize = chunksize

    def apply(self, schema: ResolvedSchema) -> ResolvedSchema:
        """Insert the ``trace`` dimension and its chunk before the vertical dimension."""
        new_dimensions: list[DimensionSpec] = []
        new_chunk_shape: list[int] = []
        for dim, chunk in zip(schema.dimensions, schema.chunk_shape, strict=True):
            if dim.is_spatial:
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk)
                continue
            new_dimensions.append(DimensionSpec(name=_TRACE_DIM, is_spatial=True, is_calculated=True))
            new_chunk_shape.append(self.chunksize)
            new_dimensions.append(dim)
            new_chunk_shape.append(chunk)

        return schema.model_copy(update={"dimensions": new_dimensions, "chunk_shape": tuple(new_chunk_shape)})


class CollapseToTraceEffect(SchemaEffect):
    """Collapse selected spatial dimensions into a single ``trace`` dimension.

    Mirrors the ``NonBinned`` override. The collapsed dimensions are re-expressed as
    non-dimension coordinates over ``trace``, and any coordinate that referenced a
    collapsed dimension is rewritten to depend on ``trace`` instead.

    Args:
        chunksize: Chunk size assigned to the inserted ``trace`` dimension.
        collapse_dims: Names of spatial dimensions to collapse. ``None`` collapses every
            spatial dimension except the first; an empty tuple collapses nothing.
    """

    def __init__(self, chunksize: int | None, collapse_dims: tuple[str, ...] | None = None) -> None:
        self.chunksize = chunksize
        self.collapse_dims = collapse_dims

    def _resolve_collapse_dims(self, schema: ResolvedSchema) -> tuple[str, ...]:
        """Resolve the effective set of dimension names to collapse for ``schema``."""
        if self.collapse_dims is not None:
            return self.collapse_dims
        spatial = schema.spatial_dimensions()
        return tuple(dim.name for dim in spatial[1:]) if len(spatial) > 1 else ()

    def apply(self, schema: ResolvedSchema) -> ResolvedSchema:
        """Collapse the configured dimensions into ``trace`` and rewrite coordinates."""
        collapse = self._resolve_collapse_dims(schema)
        collapse_set = set(collapse)

        new_dimensions: list[DimensionSpec] = []
        new_chunk_shape: list[int] = []
        replaced_count = 0
        for dim, chunk in zip(schema.dimensions, schema.chunk_shape, strict=True):
            if dim.name in collapse_set:
                replaced_count += 1
                continue
            if dim.is_spatial:
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk)
                continue
            if replaced_count > 0:
                new_dimensions.append(DimensionSpec(name=_TRACE_DIM, is_spatial=True, is_calculated=True))
                new_chunk_shape.append(self.chunksize)
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
        """Rewrite coordinate dimensions and append collapsed dims as ``trace`` coordinates."""
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
