"""Schema-driven factory for MDIO datasets."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

from mdio.builder.dataset_builder import MDIODatasetBuilder
from mdio.builder.schemas import compressors
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.core.utils_write import MAX_COORDINATES_BYTES
from mdio.core.utils_write import MAX_SIZE_LIVE_MASK
from mdio.core.utils_write import get_constrained_chunksize

if TYPE_CHECKING:
    from mdio.builder.schemas import Dataset
    from mdio.builder.schemas.dtype import StructuredType
    from mdio.builder.schemas.v1.units import AllUnitModel
    from mdio.ingestion.schema import ResolvedSchema

logger = logging.getLogger(__name__)


def _chunk_variable(ds: Dataset, target_variable_name: str) -> None:
    """Determines and sets the chunking for a specific Variable in the Dataset."""
    index = next((i for i, obj in enumerate(ds.variables) if obj.name == target_variable_name), None)
    if index is None:
        return

    def determine_target_size(var_type: str) -> int:
        """Determines the target size (in bytes) for a Variable based on its type."""
        if var_type == "bool":
            return MAX_SIZE_LIVE_MASK
        return MAX_COORDINATES_BYTES

    var_type = ds.variables[index].data_type
    full_shape = tuple(dim.size for dim in ds.variables[index].dimensions)
    target_size = determine_target_size(var_type)

    chunk_shape = get_constrained_chunksize(full_shape, var_type, target_size)
    chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=chunk_shape))

    if ds.variables[index].metadata is None:
        ds.variables[index].metadata = VariableMetadata()

    ds.variables[index].metadata.chunk_grid = chunk_grid


def _resolve_chunks(chunk_shape: tuple[int, ...], sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Resolve chunk shapes by substituting -1 with actual sizes.

    Args:
        chunk_shape: Configured chunk shape.
        sizes: Actual sizes of each dimension.

    Returns:
        tuple[int, ...]: Resolved chunk shape.
    """
    return tuple(size if chunk_size == -1 else chunk_size for chunk_size, size in zip(chunk_shape, sizes, strict=True))


def _resolve_shards(
    shard_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    sizes: tuple[int, ...],
) -> tuple[int, ...]:
    """Resolve the shard shape (``-1`` -> full size) and validate it against the chunk shape.

    An empty ``shard_shape`` means sharding is disabled and returns ``()``. Otherwise the shard
    is the Zarr v3 storage-object unit and must be a whole multiple of the (resolved) chunk shape
    along every dimension, so an integer number of chunks packs into each shard.

    Args:
        shard_shape: Configured shard shape (may contain ``-1``); ``()`` disables sharding.
        chunk_shape: Already-resolved chunk shape (no ``-1``).
        sizes: Actual sizes of each dimension.

    Returns:
        The resolved shard shape, or ``()`` when sharding is disabled.

    Raises:
        ValueError: If the shard rank differs from the chunk rank, or a shard extent is not a
            positive whole multiple of the corresponding chunk extent.
    """
    if not shard_shape:
        return ()
    if len(shard_shape) != len(chunk_shape):
        msg = f"Shard shape {shard_shape} rank does not match chunk shape {chunk_shape}."
        raise ValueError(msg)
    resolved = tuple(size if s == -1 else s for s, size in zip(shard_shape, sizes, strict=True))
    for shard_size, chunk_size in zip(resolved, chunk_shape, strict=True):
        if shard_size <= 0 or shard_size % chunk_size != 0:
            msg = (
                f"Shard shape {resolved} must be a positive whole multiple of the chunk shape "
                f"{chunk_shape} along every dimension (offending pair: shard={shard_size}, "
                f"chunk={chunk_size})."
            )
            raise ValueError(msg)
    return resolved


def _shard_grid(shard_shape: tuple[int, ...]) -> RegularChunkGrid | None:
    """Build a shard grid model from a resolved shard shape (``()`` -> ``None``)."""
    if not shard_shape:
        return None
    return RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=shard_shape))


def _create_dataset_builder(schema: ResolvedSchema) -> MDIODatasetBuilder:
    """Create and initialize the MDIODatasetBuilder with attributes.

    Args:
        schema: Resolved schema.

    Returns:
        MDIODatasetBuilder: The initialized dataset builder.
    """
    attributes = dict(schema.metadata) if schema.metadata else {}
    attributes["defaultVariableName"] = schema.default_variable_name

    return MDIODatasetBuilder(name=schema.name, attributes=attributes)


def _add_dimensions_and_coordinates(
    builder: MDIODatasetBuilder,
    schema: ResolvedSchema,
    sizes: tuple[int, ...],
    units: dict[str, AllUnitModel],
) -> None:
    """Add dimensions and coordinates to the builder.

    Args:
        builder: MDIO dataset builder.
        schema: Resolved schema.
        sizes: Actual sizes of each dimension.
        units: Dictionary mapping coordinate/dimension names to AllUnitModel.
    """
    for dim_spec, size in zip(schema.dimensions, sizes, strict=True):
        builder.add_dimension(dim_spec.name, size)

    for dim_spec in schema.dimensions:
        if dim_spec.is_calculated and dim_spec.name != "trace":
            continue
        builder.add_coordinate(
            name=dim_spec.name,
            dimensions=(dim_spec.name,),
            data_type=dim_spec.dtype,
            metadata=CoordinateMetadata(units_v1=units.get(dim_spec.name)),
        )

    compressor = compressors.Blosc(cname=compressors.BloscCname.zstd)
    for coord_spec in schema.coordinates:
        builder.add_coordinate(
            name=coord_spec.name,
            dimensions=coord_spec.dimensions,
            data_type=coord_spec.dtype,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=units.get(coord_spec.name)),
        )


def _add_trace_mask_and_headers(
    builder: MDIODatasetBuilder,
    schema: ResolvedSchema,
    resolved_chunks: tuple[int, ...],
    header_dtype: StructuredType | None,
) -> None:
    """Add trace mask and headers variables to the builder.

    Args:
        builder: MDIO dataset builder.
        schema: Resolved schema.
        resolved_chunks: Resolved chunk shapes.
        header_dtype: Structured dtype for trace headers.
    """
    compressor = compressors.Blosc(cname=compressors.BloscCname.zstd)
    spatial_dim_names = tuple(dim.name for dim in schema.dimensions if dim.is_spatial)
    coordinate_names = [coord.name for coord in schema.coordinates]

    builder.add_variable(
        name="trace_mask",
        dimensions=spatial_dim_names,
        data_type=ScalarType.BOOL,
        compressor=compressor,
        coordinates=coordinate_names,
    )

    if header_dtype is not None:
        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=resolved_chunks[:-1]))
        # NOTE: headers are intentionally NOT sharded. They use a structured (void) dtype, and
        # Zarr's sharding partial-encode path tries to hash the void fill value
        # (TypeError: unhashable 'writeable void-scalar'). Headers are also tiny relative to the
        # sample cube, so sharding them has little upside. Sharding applies to the data variable.
        builder.add_variable(
            name="headers",
            dimensions=spatial_dim_names,
            data_type=header_dtype,
            compressor=compressor,
            coordinates=coordinate_names,
            metadata=VariableMetadata(chunk_grid=chunk_grid),
        )


def _add_main_and_extra_variables(
    builder: MDIODatasetBuilder,
    schema: ResolvedSchema,
    resolved_chunks: tuple[int, ...],
    resolved_shards: tuple[int, ...],
    units: dict[str, AllUnitModel],
    extra_variables: list[dict[str, Any]],
) -> None:
    """Add main data and extra variables to the builder.

    Args:
        builder: MDIO dataset builder.
        schema: Resolved schema.
        resolved_chunks: Resolved chunk shapes.
        resolved_shards: Resolved shard shapes (``()`` disables sharding).
        units: Dictionary mapping coordinate/dimension names to AllUnitModel.
        extra_variables: Optional list of additional variables.
    """
    compressor = compressors.Blosc(cname=compressors.BloscCname.zstd)
    coordinate_names = [coord.name for coord in schema.coordinates]

    chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=resolved_chunks))
    builder.add_variable(
        name=schema.default_variable_name,
        dimensions=tuple(dim.name for dim in schema.dimensions),
        data_type=ScalarType.FLOAT32,
        compressor=compressor,
        coordinates=coordinate_names,
        metadata=VariableMetadata(
            chunk_grid=chunk_grid,
            shard_grid=_shard_grid(resolved_shards),
            units_v1=units.get(schema.default_variable_name),
        ),
    )

    for var_dict in extra_variables:
        builder.add_variable(**var_dict)


def _apply_dynamic_chunking(ds: Dataset, coordinate_names: list[str]) -> None:
    """Dynamically chunk trace_mask and coordinates based on their data types and sizes.

    Args:
        ds: The dataset model.
        coordinate_names: List of coordinate names.
    """
    _chunk_variable(ds=ds, target_variable_name="trace_mask")
    for coord_name in coordinate_names:
        _chunk_variable(ds=ds, target_variable_name=coord_name)


def build_mdio_dataset(
    schema: ResolvedSchema,
    sizes: tuple[int, ...],
    header_dtype: StructuredType | None = None,
    units: dict[str, AllUnitModel] | None = None,
    extra_variables: list[dict[str, Any]] | None = None,
) -> Dataset:
    """Build an MDIO Dataset model purely from a ResolvedSchema and dimensions sizes.

    Args:
        schema: Resolved schema describing dimensions, coordinates, metadata, etc.
        sizes: Actual sizes of each dimension.
        header_dtype: Structured dtype for trace headers.
        units: Dictionary mapping coordinate/dimension names to AllUnitModel.
        extra_variables: Optional list of additional variables to add (e.g. raw_headers).

    Returns:
        Dataset: The completed Dataset schema model.
    """
    units = units or {}
    extra_variables = extra_variables or []

    resolved_chunks = _resolve_chunks(schema.chunk_shape, sizes)
    resolved_shards = _resolve_shards(schema.shard_shape, resolved_chunks, sizes)
    builder = _create_dataset_builder(schema)

    _add_dimensions_and_coordinates(
        builder=builder,
        schema=schema,
        sizes=sizes,
        units=units,
    )

    _add_trace_mask_and_headers(
        builder=builder,
        schema=schema,
        resolved_chunks=resolved_chunks,
        header_dtype=header_dtype,
    )

    _add_main_and_extra_variables(
        builder=builder,
        schema=schema,
        resolved_chunks=resolved_chunks,
        resolved_shards=resolved_shards,
        units=units,
        extra_variables=extra_variables,
    )

    ds = builder.build()

    coordinate_names = [coord.name for coord in schema.coordinates]
    _apply_dynamic_chunking(ds=ds, coordinate_names=coordinate_names)

    return ds
