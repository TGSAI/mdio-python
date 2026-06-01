"""Schema resolution: turn a template + grid overrides into a final, ingestion-ready schema."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from mdio.builder.templates.types import CoordinateSpec  # noqa: TC001  (pydantic needs this at runtime)

if TYPE_CHECKING:
    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.segy.geometry import GridOverrides


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
    """

    name: str
    is_spatial: bool = True
    is_calculated: bool = False


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

    def required_header_fields(self) -> set[str]:
        """Names that must be readable from SEG-Y trace headers to materialize this schema."""
        fields = {dim.name for dim in self.dimensions if dim.is_spatial and not dim.is_calculated}
        fields.update(coord.name for coord in self.coordinates)
        # coordinate_scalar is always needed to scale X/Y coordinates.
        fields.add("coordinate_scalar")
        return fields

    def spatial_dimensions(self) -> list[DimensionSpec]:
        """Get only spatial dimensions (excludes vertical/trace domain)."""
        return [dim for dim in self.dimensions if dim.is_spatial]


class SchemaResolver:
    """Resolves template + grid overrides into a final schema.

    This class takes a template and optional grid overrides and produces
    a ResolvedSchema that completely specifies the dataset structure before
    any data is scanned or processed.
    """

    def resolve(
        self,
        template: AbstractDatasetTemplate,
        grid_overrides: GridOverrides | None = None,
    ) -> ResolvedSchema:
        """Resolve template and overrides into final schema.

        Args:
            template: The MDIO dataset template
            grid_overrides: Optional grid override configuration

        Returns:
            ResolvedSchema with all dimensions, coordinates, and metadata resolved
        """
        schema = self._template_to_schema(template)

        if grid_overrides:
            schema = self._apply_override_transformations(schema, grid_overrides)

        return schema

    def _template_to_schema(self, template: AbstractDatasetTemplate) -> ResolvedSchema:
        """Convert a template to a resolved schema without overrides."""
        calculated = set(template.calculated_dimension_names)
        dimensions = [
            DimensionSpec(name=name, is_spatial=True, is_calculated=name in calculated)
            for name in template.spatial_dimension_names
        ]
        dimensions.append(DimensionSpec(name=template.dimension_names[-1], is_spatial=False))

        return ResolvedSchema(
            name=template.name,
            dimensions=dimensions,
            coordinates=list(template.declare_coordinate_specs()),
            chunk_shape=template.full_chunk_shape,
            metadata=template._load_dataset_attributes() or {},
            default_variable_name=template.default_variable_name,
        )

    def _apply_override_transformations(
        self,
        schema: ResolvedSchema,
        grid_overrides: GridOverrides,
    ) -> ResolvedSchema:
        """Apply grid override transformations to the schema."""
        schema_dict = schema.model_dump()

        if grid_overrides.non_binned:
            schema_dict = self._apply_non_binned_transform(schema_dict, grid_overrides)
        elif grid_overrides.has_duplicates:
            schema_dict = self._apply_duplicate_transform(schema_dict)

        schema_dict["metadata"]["gridOverrides"] = grid_overrides.to_legacy_dict()

        return ResolvedSchema(**schema_dict)

    def _apply_non_binned_transform(
        self,
        schema_dict: dict,
        grid_overrides: GridOverrides,
    ) -> dict:
        """Replace selected spatial dimensions with a single ``trace`` dimension."""
        dimensions = schema_dict["dimensions"]
        chunk_shape = list(schema_dict["chunk_shape"])

        replace_dims = grid_overrides.non_binned_dims
        if replace_dims is None:
            # Default: replace all spatial dims except the first.
            spatial_dims = [d for d in dimensions if d["is_spatial"]]
            replace_dims = [d["name"] for d in spatial_dims[1:]] if len(spatial_dims) > 1 else []

        new_dimensions = []
        new_chunk_shape = []
        replaced_count = 0

        for i, dim in enumerate(dimensions):
            if dim["name"] in replace_dims:
                replaced_count += 1
                continue
            if dim["is_spatial"]:
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk_shape[i])
            else:
                if replaced_count > 0:
                    new_dimensions.append(DimensionSpec(name="trace", is_spatial=True, is_calculated=True).model_dump())
                    new_chunk_shape.append(grid_overrides.chunksize)
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk_shape[i])

        schema_dict["dimensions"] = new_dimensions
        schema_dict["chunk_shape"] = tuple(new_chunk_shape)

        # Rewrite coordinate dimension references: collapsed dims drop out, replaced by ``trace``.
        replaced_dims_set = set(replace_dims)
        updated_coordinates = []
        for coord in schema_dict["coordinates"]:
            original_dims = coord["dimensions"]
            had_collapsed_dims = any(d in replaced_dims_set for d in original_dims)
            coord_dims = [d for d in original_dims if d not in replaced_dims_set]

            if had_collapsed_dims and replaced_count > 0:
                coord_dims.append("trace")

            updated_coord = dict(coord)
            updated_coord["dimensions"] = tuple(coord_dims)
            updated_coordinates.append(updated_coord)
        schema_dict["coordinates"] = updated_coordinates

        return schema_dict

    def _apply_duplicate_transform(self, schema_dict: dict) -> dict:
        """Insert a ``trace`` dimension with chunksize 1 before the vertical dimension."""
        dimensions = schema_dict["dimensions"]
        chunk_shape = list(schema_dict["chunk_shape"])

        new_dimensions = []
        new_chunk_shape = []

        for i, dim in enumerate(dimensions):
            if dim["is_spatial"]:
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk_shape[i])
            else:
                new_dimensions.append(DimensionSpec(name="trace", is_spatial=True, is_calculated=True).model_dump())
                new_chunk_shape.append(1)
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk_shape[i])

        schema_dict["dimensions"] = new_dimensions
        schema_dict["chunk_shape"] = tuple(new_chunk_shape)

        return schema_dict
