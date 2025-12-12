"""Dataset Factory for MDIO Ingestion.

This module provides a factory for building MDIO datasets from resolved schemas
and dimensions. It centralizes all dataset construction logic in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdio.builder.dataset_builder import MDIODatasetBuilder
from mdio.builder.schemas import compressors
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.core.utils_write import MAX_COORDINATES_BYTES
from mdio.core.utils_write import MAX_SIZE_LIVE_MASK
from mdio.core.utils_write import get_constrained_chunksize

if TYPE_CHECKING:
    from mdio.builder.schemas.dtype import StructuredType
    from mdio.builder.schemas.v1.dataset import Dataset
    from mdio.core.dimension import Dimension
    from mdio.ingestion.schema_resolver import ResolvedSchema


class DatasetFactory:
    """Factory for building MDIO datasets from schemas.

    This class takes a resolved schema and dimensions and builds
    a complete MDIO dataset with all variables and coordinates.
    """

    def build(
        self,
        schema: ResolvedSchema,
        dimensions: list[Dimension],
        header_dtype: StructuredType | None = None,
        include_raw_headers: bool = False,
    ) -> Dataset:
        """Build MDIO dataset from schema and dimensions.

        Args:
            schema: Resolved schema specifying dataset structure
            dimensions: List of dimension objects with coordinates
            header_dtype: Optional structured type for trace headers
            include_raw_headers: Whether to include raw binary headers (Zarr v3 only)

        Returns:
            Complete Dataset ready for xarray conversion
        """
        # Create dimension sizes dict
        dim_sizes = {dim.name: len(dim.coords) for dim in dimensions}

        # Initialize builder
        schema.metadata["defaultVariableName"] = schema.default_variable_name
        builder = MDIODatasetBuilder(
            name=schema.name,
            attributes=schema.metadata,
        )

        # Add dimensions
        for dim in dimensions:
            builder.add_dimension(dim.name, len(dim.coords))

        # Add dimension coordinates
        self._add_dimension_coordinates(builder, schema, dimensions)

        # Add non-dimension coordinates
        self._add_non_dimension_coordinates(builder, schema, dim_sizes)

        # Add main data variable
        self._add_data_variable(builder, schema, dimensions)

        # Add trace mask
        self._add_trace_mask(builder, schema, dim_sizes)

        # Add trace headers if requested
        if header_dtype is not None:
            self._add_trace_headers(builder, schema, dim_sizes, header_dtype)

        # Add raw headers if requested
        if include_raw_headers:
            self._add_raw_headers(builder, schema, dim_sizes)

        return builder.build()

    def _add_dimension_coordinates(
        self,
        builder: MDIODatasetBuilder,
        schema: ResolvedSchema,
        dimensions: list[Dimension],
    ) -> None:
        """Add dimension coordinate variables."""
        for dim in dimensions:
            # Find the dimension spec to get metadata
            dim_spec = next((d for d in schema.dimensions if d.name == dim.name), None)

            # Determine dtype based on dimension name
            if dim.name in ("time", "depth") or "trace" in dim.name:
                dtype = ScalarType.INT32
            else:
                dtype = ScalarType.INT32

            builder.add_coordinate(
                name=dim.name,
                dimensions=(dim.name,),
                data_type=dtype,
                metadata=VariableMetadata(units_v1=None),  # Units will be set by template
            )

    def _add_non_dimension_coordinates(
        self,
        builder: MDIODatasetBuilder,
        schema: ResolvedSchema,
        dim_sizes: dict[str, int],
    ) -> None:
        """Add non-dimension coordinate variables."""
        for coord_spec in schema.coordinates:
            # Compute chunk shape for this coordinate
            coord_dim_sizes = tuple(dim_sizes[dim] for dim in coord_spec.dimensions)
            coord_chunk_shape = get_constrained_chunksize(
                coord_dim_sizes,
                coord_spec.dtype,
                MAX_COORDINATES_BYTES,
            )
            chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=coord_chunk_shape))

            builder.add_coordinate(
                name=coord_spec.name,
                dimensions=coord_spec.dimensions,
                data_type=coord_spec.dtype,
                compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),
                metadata=VariableMetadata(units_v1=None, chunk_grid=chunk_grid),
            )

    def _add_data_variable(
        self,
        builder: MDIODatasetBuilder,
        schema: ResolvedSchema,
        dimensions: list[Dimension],
    ) -> None:
        """Add main data variable (amplitude/traces)."""
        dim_names = tuple(dim.name for dim in dimensions)
        coord_names = tuple(coord.name for coord in schema.coordinates)

        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=schema.chunk_shape))

        builder.add_variable(
            name=schema.default_variable_name,
            dimensions=dim_names,
            data_type=ScalarType.FLOAT32,
            compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),
            coordinates=coord_names,
            metadata=VariableMetadata(chunk_grid=chunk_grid, units_v1=None),
        )

    def _add_trace_mask(
        self,
        builder: MDIODatasetBuilder,
        schema: ResolvedSchema,
        dim_sizes: dict[str, int],
    ) -> None:
        """Add trace mask variable."""
        spatial_dims = tuple(d.name for d in schema.spatial_dimensions())
        spatial_sizes = tuple(dim_sizes[dim] for dim in spatial_dims)
        coord_names = tuple(coord.name for coord in schema.coordinates)

        mask_chunk_shape = get_constrained_chunksize(
            spatial_sizes,
            ScalarType.BOOL,
            MAX_SIZE_LIVE_MASK,
        )
        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=mask_chunk_shape))

        builder.add_variable(
            name="trace_mask",
            dimensions=spatial_dims,
            data_type=ScalarType.BOOL,
            compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),
            coordinates=coord_names,
            metadata=VariableMetadata(chunk_grid=chunk_grid),
        )

    def _add_trace_headers(
        self,
        builder: MDIODatasetBuilder,
        schema: ResolvedSchema,
        dim_sizes: dict[str, int],
        header_dtype: StructuredType,
    ) -> None:
        """Add trace headers variable."""
        spatial_dims = tuple(d.name for d in schema.spatial_dimensions())
        coord_names = tuple(coord.name for coord in schema.coordinates)

        # Use spatial chunk shape (no vertical dimension)
        spatial_chunk_shape = tuple(
            chunk for chunk, dim in zip(schema.chunk_shape, schema.dimensions) if dim.is_spatial
        )
        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=spatial_chunk_shape))

        builder.add_variable(
            name="headers",
            dimensions=spatial_dims,
            data_type=header_dtype,
            compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),
            coordinates=coord_names,
            metadata=VariableMetadata(chunk_grid=chunk_grid),
        )

    def _add_raw_headers(
        self,
        builder: MDIODatasetBuilder,
        schema: ResolvedSchema,
        dim_sizes: dict[str, int],
    ) -> None:
        """Add raw binary headers variable (Zarr v3 only)."""
        spatial_dims = tuple(d.name for d in schema.spatial_dimensions())

        # Use spatial chunk shape (no vertical dimension)
        spatial_chunk_shape = tuple(
            chunk for chunk, dim in zip(schema.chunk_shape, schema.dimensions) if dim.is_spatial
        )
        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=spatial_chunk_shape))

        builder.add_variable(
            name="raw_headers",
            long_name="Raw Binary Trace Headers",
            dimensions=spatial_dims,
            data_type=ScalarType.BYTES240,
            compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),
            coordinates=None,
            metadata=VariableMetadata(chunk_grid=chunk_grid),
        )
