"""Factory implementation for MDIO v1 datasets."""

from datetime import datetime
from datetime import timezone
from typing import List
from typing import Optional

from mdio.schema.compressors import ZFP
from mdio.schema.compressors import Blosc
from mdio.schema.dimension import NamedDimension
from mdio.schema.dtype import ScalarType
from mdio.schema.dtype import StructuredType
from mdio.schema.metadata import UserAttributes
from mdio.schema.v1.dataset import Dataset
from mdio.schema.v1.units import AllUnits
from mdio.schema.v1.variable import Coordinate
from mdio.schema.v1.variable import Variable
from mdio.schema.v1.variable import VariableMetadata

from ._serializer import (
    make_coordinate,
    make_dataset,
    make_dataset_metadata,
    make_named_dimension,
    make_variable,
)


class AbstractTemplateFactory:
    """Abstract factory for creating MDIO datasets."""

    def __init__(self, name: str):
        """Initialize the factory.

        Args:
            name: Name of the dataset
        """
        self.name = name
        self.api_version = "1.0.0"  # TODO: Pull from package metadata
        self.created_on = datetime.now(timezone.utc)
        self.dimensions: List[NamedDimension] = []
        self.coordinates: List[Coordinate] = []
        self.variables: List[Variable] = []

    def add_dimension(self, name: str, size: int) -> "AbstractTemplateFactory":
        """Add a dimension to the factory."""
        self.dimensions.append(make_named_dimension(name, size))
        return self

    def add_coordinate(
        self,
        name: str = "",
        dimensions: Optional[List[NamedDimension | str]] = None,
        data_type: ScalarType | StructuredType = ScalarType.FLOAT32,
        metadata: Optional[List[AllUnits | UserAttributes]] = None,
    ) -> "AbstractTemplateFactory":
        """Add a coordinate to the factory."""
        if name == "":
            name = f"coord_{len(self.coordinates)}"
        if dimensions is None:
            dimensions = self.dimensions
        self.coordinates.append(make_coordinate(name, dimensions, data_type, metadata))
        return self

    def add_variable(
        self,
        name: str = "",
        dimensions: Optional[List[NamedDimension | str]] = None,
        data_type: ScalarType | StructuredType = ScalarType.FLOAT32,
        compressor: Blosc | ZFP | None = None,
        coordinates: Optional[List[Coordinate | str]] = None,
        metadata: Optional[VariableMetadata] = None,
    ) -> "AbstractTemplateFactory":
        """Add a variable to the factory."""
        if name == "":
            name = f"var_{len(self.variables)}"
        if dimensions is None:
            dimensions = self.dimensions
        self.variables.append(
            make_variable(
                name, dimensions, data_type, compressor, coordinates, metadata
            )
        )
        return self

    def _compose_metadata(self):
        """Compose the DatasetMetadata with the given name, api_version, and created_on."""
        return make_dataset_metadata(self.name, self.api_version, self.created_on)

    def _compose_variables(self) -> List[Variable]:
        """Compose the Variables with the given parameters."""
        return [
            make_variable(
                self.name,
                self.dimensions,
                self.data_type,
                self.compressor,
                self.coordinates,
                self.metadata,
            )
        ]

    def make_dataset(self, variables: List[Variable]) -> Dataset:
        """Create a Dataset with the given variables and metadata."""
        return Dataset(variables=variables, metadata=self._compose_metadata())
