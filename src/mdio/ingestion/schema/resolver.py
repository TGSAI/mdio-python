"""Schema resolution: turn a template (and an optional reshape) into an ingestion-ready schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdio.builder.schemas.dtype import ScalarType
from mdio.ingestion.schema.models import DimensionSpec
from mdio.ingestion.schema.models import ResolvedSchema

if TYPE_CHECKING:
    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.ingestion.schema.models import SchemaEffect


class SchemaResolver:
    """Resolves a template (and an optional layout effect) into a final schema.

    This class converts a template into a :class:`ResolvedSchema` that completely specifies
    the dataset structure before any data is scanned or processed. It is format-agnostic: it
    knows nothing about SEG-Y or grid overrides. A caller that needs to reshape the layout
    (e.g. to insert or collapse a ``trace`` dimension) selects the appropriate
    :class:`~mdio.ingestion.schema.models.SchemaEffect` and passes it in; the resolver only
    applies it. For SEG-Y, that selection is owned by
    :class:`~mdio.ingestion.segy.index_strategies.IndexStrategyRegistry`.
    """

    def resolve(
        self,
        template: AbstractDatasetTemplate,
        effect: SchemaEffect | None = None,
    ) -> ResolvedSchema:
        """Resolve a template and optional layout effect into a final schema.

        Args:
            template: The MDIO dataset template.
            effect: Optional layout reshape to apply to the template-derived schema.

        Returns:
            ResolvedSchema with all dimensions, coordinates, and metadata resolved.
        """
        schema = self._template_to_schema(template)
        if effect is not None:
            schema = effect.apply(schema)
        return schema

    def _template_to_schema(self, template: AbstractDatasetTemplate) -> ResolvedSchema:
        """Convert a template to a resolved schema without overrides."""
        calculated = set(template.calculated_dimension_names)
        dim_dtypes = template.declare_dim_coordinate_types()
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
