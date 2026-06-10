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


class SchemaEffect(ABC):
    """A format-agnostic reshape applied to a :class:`ResolvedSchema`.

    Concrete effects and the logic that selects them live with the ingestion format that
    needs them (for SEG-Y grid overrides, see :mod:`mdio.ingestion.segy.schema_effects`).
    """

    @abstractmethod
    def apply(self, schema: ResolvedSchema) -> ResolvedSchema:
        """Return a new schema with this effect applied; ``schema`` is left unchanged."""
