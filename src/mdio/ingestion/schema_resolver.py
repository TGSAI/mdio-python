"""Schema Resolution System for MDIO Ingestion.

This module resolves the final dataset schema from a template and grid overrides
before any data is scanned. This allows for early validation and clear separation
between configuration and execution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from mdio.builder.schemas.dtype import ScalarType

if TYPE_CHECKING:
    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.segy.geometry import GridOverrides


class DimensionSpec(BaseModel):
    """Specification for a dimension in the final dataset.

    Attributes:
        name: Name of the dimension (e.g., "shot_point", "cable", "trace", "time")
        source: Where this dimension comes from:
            - "header": Read from SEG-Y trace headers
            - "computed": Computed from other headers (e.g., shot_index from shot_point)
            - "synthetic": Generated (e.g., time/depth axis from binary header)
        header_key: The SEG-Y header field name if source is "header" or "computed"
        is_spatial: Whether this is a spatial dimension (not vertical domain)
    """

    name: str
    source: Literal["header", "computed", "synthetic"]
    header_key: str | None = None
    is_spatial: bool = True


class CoordinateSpec(BaseModel):
    """Specification for a coordinate in the final dataset.

    Attributes:
        name: Name of the coordinate (e.g., "cdp_x", "gun", "source_coord_x")
        dimensions: Tuple of dimension names this coordinate depends on
        dtype: Data type for the coordinate
        source: Where this coordinate comes from:
            - "header": Read from SEG-Y trace headers
            - "computed": Computed from headers with a function
        header_key: The SEG-Y header field name if source is "header"
        computation: Optional function to compute coordinate from headers
    """

    name: str
    dimensions: tuple[str, ...]
    dtype: ScalarType
    source: Literal["header", "computed"]
    header_key: str | None = None
    computation: Callable | None = Field(default=None, exclude=True)


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
        """Get all header fields required for this schema."""
        fields = set()

        # Add dimension header keys
        for dim in self.dimensions:
            if dim.header_key is not None:
                fields.add(dim.header_key)

        # Add coordinate header keys
        for coord in self.coordinates:
            if coord.header_key is not None:
                fields.add(coord.header_key)

        # Always need coordinate_scalar for X/Y coordinates
        fields.add("coordinate_scalar")

        return fields

    def spatial_dimensions(self) -> list[DimensionSpec]:
        """Get only spatial dimensions (excludes vertical/trace domain)."""
        return [dim for dim in self.dimensions if dim.is_spatial]

    def computed_dimensions(self) -> list[DimensionSpec]:
        """Get dimensions that need to be computed from headers."""
        return [dim for dim in self.dimensions if dim.source == "computed"]


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
        # Start with base schema from template
        schema = self._template_to_schema(template)

        # Apply grid override transformations if present
        if grid_overrides and grid_overrides:  # Check __bool__
            schema = self._apply_override_transformations(schema, grid_overrides)

        return schema

    def _template_to_schema(self, template: AbstractDatasetTemplate) -> ResolvedSchema:
        """Convert a template to a resolved schema without overrides."""
        # Extract dimensions from template
        dimensions = []
        spatial_dims = template.spatial_dimension_names
        vertical_dim = template.dimension_names[-1]

        for dim_name in spatial_dims:
            # Determine if this is computed (like shot_index)
            is_computed = dim_name in template.calculated_dimension_names
            source = "computed" if is_computed else "header"

            dimensions.append(
                DimensionSpec(
                    name=dim_name,
                    source=source,
                    header_key=dim_name,  # Usually same as name
                    is_spatial=True,
                )
            )

        # Add vertical dimension (time/depth)
        dimensions.append(
            DimensionSpec(
                name=vertical_dim,
                source="synthetic",
                header_key=None,
                is_spatial=False,
            )
        )

        # Extract coordinates from template
        coordinates = []

        # Physical coordinates (cdp_x, cdp_y, source_coord_x, etc.)
        for coord_name in template.physical_coordinate_names:
            # Physical coordinates are typically over spatial dimensions only
            # Need to determine dimensionality from template specifics
            coord_dims = self._infer_coordinate_dimensions(template, coord_name)

            coordinates.append(
                CoordinateSpec(
                    name=coord_name,
                    dimensions=coord_dims,
                    dtype=ScalarType.FLOAT64,
                    source="header",
                    header_key=coord_name,
                )
            )

        # Logical coordinates (gun, etc.)
        for coord_name in template.logical_coordinate_names:
            coord_dims = self._infer_coordinate_dimensions(template, coord_name)

            coordinates.append(
                CoordinateSpec(
                    name=coord_name,
                    dimensions=coord_dims,
                    dtype=ScalarType.UINT8 if coord_name == "gun" else ScalarType.INT32,
                    source="header",
                    header_key=coord_name,
                )
            )

        return ResolvedSchema(
            name=template.name,
            dimensions=dimensions,
            coordinates=coordinates,
            chunk_shape=template.full_chunk_shape,
            metadata=template._load_dataset_attributes() or {},
            default_variable_name=template.default_variable_name,
        )

    def _infer_coordinate_dimensions(
        self,
        template: AbstractDatasetTemplate,
        coord_name: str,
    ) -> tuple[str, ...]:
        """Infer coordinate dimensionality from template type.

        This is template-specific logic that determines which dimensions
        a coordinate depends on.
        """
        spatial_dims = template.spatial_dimension_names
        template_name = template.name.lower()

        # CDP coordinates are over inline/crossline for 3D, over cdp for 2D
        if coord_name in ("cdp_x", "cdp_y"):
            if "2d" in template_name:
                return ("cdp",)
            if "3d" in template_name:
                # For CDP gathers, it's just inline/crossline
                # For other 3D templates, might be different
                if "cdp" in template_name.lower():
                    return ("inline", "crossline")
                # For COCA and others, also inline/crossline
                return ("inline", "crossline")

        # Source coordinates are over shot_point for streamer
        if coord_name in ("source_coord_x", "source_coord_y"):
            return ("shot_point",)

        # Group coordinates are over all spatial dims for streamer
        if coord_name in ("group_coord_x", "group_coord_y"):
            return spatial_dims

        # Gun is over shot_point
        if coord_name == "gun":
            return ("shot_point",)

        # Default: over all spatial dimensions
        return spatial_dims

    def _apply_override_transformations(
        self,
        schema: ResolvedSchema,
        grid_overrides: GridOverrides,
    ) -> ResolvedSchema:
        """Apply grid override transformations to the schema.

        Args:
            schema: Base schema from template
            grid_overrides: Grid overrides to apply

        Returns:
            Transformed schema
        """
        # Clone schema for modification
        schema_dict = schema.model_dump()

        # Apply NonBinned transformation
        if grid_overrides.non_binned:
            schema_dict = self._apply_non_binned_transform(schema_dict, grid_overrides)

        # Apply HasDuplicates transformation
        elif grid_overrides.has_duplicates:
            schema_dict = self._apply_duplicate_transform(schema_dict)

        # Update metadata with grid overrides
        schema_dict["metadata"]["gridOverrides"] = (
            grid_overrides.model_dump(by_alias=True, exclude_defaults=True, exclude={"extra_params"})
            | grid_overrides.extra_params
        )

        return ResolvedSchema(**schema_dict)

    def _apply_non_binned_transform(
        self,
        schema_dict: dict,
        grid_overrides: GridOverrides,
    ) -> dict:
        """Transform schema for non-binned indexing.

        Replaces specified dimensions with a single "trace" dimension.
        """
        dimensions = schema_dict["dimensions"]
        chunk_shape = list(schema_dict["chunk_shape"])

        # Determine which dimensions to replace
        replace_dims = grid_overrides.replace_dims
        if replace_dims is None:
            # Default: replace all spatial dims except the first
            spatial_dims = [d for d in dimensions if d["is_spatial"]]
            if len(spatial_dims) > 1:
                replace_dims = [d["name"] for d in spatial_dims[1:]]
            else:
                replace_dims = []

        # Build new dimension list
        new_dimensions = []
        new_chunk_shape = []
        replaced_count = 0

        for i, dim in enumerate(dimensions):
            if dim["name"] in replace_dims:
                replaced_count += 1
                # Skip this dimension (will be replaced)
                continue
            if dim["is_spatial"]:
                # Keep this spatial dimension
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk_shape[i])
            else:
                # This is the vertical dimension, add trace before it
                if replaced_count > 0:
                    new_dimensions.append(
                        DimensionSpec(
                            name="trace",
                            source="computed",
                            header_key=None,
                            is_spatial=True,
                        ).model_dump()
                    )
                    new_chunk_shape.append(grid_overrides.chunksize)

                # Then add vertical dimension
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk_shape[i])

        schema_dict["dimensions"] = new_dimensions
        schema_dict["chunk_shape"] = tuple(new_chunk_shape)

        # Update coordinate dimensions to remove references to collapsed dimensions
        replaced_dims_set = set(replace_dims)
        updated_coordinates = []
        for coord in schema_dict["coordinates"]:
            original_dims = coord["dimensions"]
            # Check if this coordinate referenced any collapsed dimensions
            had_collapsed_dims = any(d in replaced_dims_set for d in original_dims)
            # Filter out collapsed dimensions
            coord_dims = list(d for d in original_dims if d not in replaced_dims_set)

            # If we removed any dimensions and trace was added, append trace
            if had_collapsed_dims and replaced_count > 0:
                coord_dims.append("trace")

            updated_coord = dict(coord)
            updated_coord["dimensions"] = tuple(coord_dims)
            updated_coordinates.append(updated_coord)
        schema_dict["coordinates"] = updated_coordinates

        return schema_dict

    def _apply_duplicate_transform(self, schema_dict: dict) -> dict:
        """Transform schema for duplicate index handling.

        Adds a "trace" dimension with chunksize=1.
        """
        dimensions = schema_dict["dimensions"]
        chunk_shape = list(schema_dict["chunk_shape"])

        # Insert trace dimension before vertical (last) dimension
        new_dimensions = []
        new_chunk_shape = []

        for i, dim in enumerate(dimensions):
            if dim["is_spatial"]:
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk_shape[i])
            else:
                # Add trace dimension before vertical
                new_dimensions.append(
                    DimensionSpec(
                        name="trace",
                        source="computed",
                        header_key=None,
                        is_spatial=True,
                    ).model_dump()
                )
                new_chunk_shape.append(1)

                # Then add vertical dimension
                new_dimensions.append(dim)
                new_chunk_shape.append(chunk_shape[i])

        schema_dict["dimensions"] = new_dimensions
        schema_dict["chunk_shape"] = tuple(new_chunk_shape)

        return schema_dict
