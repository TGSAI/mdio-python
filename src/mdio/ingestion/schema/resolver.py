"""Schema resolution: turn a template + grid overrides into a final, ingestion-ready schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdio.builder.schemas.dtype import ScalarType
from mdio.ingestion.schema.models import DimensionSpec
from mdio.ingestion.schema.models import ResolvedSchema
from mdio.ingestion.segy.index_strategies import IndexStrategyRegistry

if TYPE_CHECKING:
    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.segy.geometry import GridOverrides


class SchemaResolver:
    """Resolves template + grid overrides into a final schema.

    This class takes a template and optional grid overrides and produces a
    :class:`ResolvedSchema` that completely specifies the dataset structure before any data
    is scanned or processed. All override semantics (which dimensions collapse, where a
    ``trace`` dimension is inserted) are owned by
    :class:`~mdio.ingestion.segy.index_strategies.IndexStrategyRegistry`; this resolver only
    applies the resulting :class:`~mdio.ingestion.schema.models.SchemaEffect`.
    """

    def __init__(self) -> None:
        self._registry = IndexStrategyRegistry()

    def resolve(
        self,
        template: AbstractDatasetTemplate,
        grid_overrides: GridOverrides | None = None,
    ) -> ResolvedSchema:
        """Resolve template and overrides into final schema.

        Args:
            template: The MDIO dataset template.
            grid_overrides: Optional grid override configuration.

        Returns:
            ResolvedSchema with all dimensions, coordinates, and metadata resolved.
        """
        schema = self._template_to_schema(template)

        if not grid_overrides:
            return schema

        # Grid-override provenance is attached to the dataset at assembly time
        # (mdio.ingestion.metadata.add_grid_override_to_metadata); the resolver only
        # reshapes the schema, keeping it override-mechanics-only.
        effect = self._registry.schema_effect(grid_overrides)
        if effect is not None:
            schema = effect.apply(schema)

        return schema

    def _template_to_schema(self, template: AbstractDatasetTemplate) -> ResolvedSchema:
        """Convert a template to a resolved schema without overrides."""
        calculated = set(template.calculated_dimension_names)
        dim_dtypes = template.declare_dimension_specs()
        dimensions = [
            DimensionSpec(
                name=name,
                is_spatial=True,
                is_calculated=name in calculated,
                dtype=dim_dtypes.get(name, ScalarType.INT32),
            )
            for name in template.spatial_dimension_names
        ]
        vertical_name = template.dimension_names[-1]
        dimensions.append(
            DimensionSpec(
                name=vertical_name,
                is_spatial=False,
                dtype=dim_dtypes.get(vertical_name, ScalarType.INT32),
            )
        )

        return ResolvedSchema(
            name=template.name,
            dimensions=dimensions,
            coordinates=list(template.declare_coordinate_specs()),
            chunk_shape=template.full_chunk_shape,
            metadata=template._load_dataset_attributes() or {},
            default_variable_name=template.default_variable_name,
        )
