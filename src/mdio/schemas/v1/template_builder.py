"""Builder pattern implementation for MDIO v1 schema models."""

from datetime import datetime, timezone
from typing import Any, Optional, List, Dict, Union
from enum import Enum, auto

from pydantic import AwareDatetime

from mdio.schema.dimension import NamedDimension
from mdio.schema.compressors import Blosc, ZFP
from mdio.schema.dtype import ScalarType, StructuredType
from mdio.schema.metadata import UserAttributes
from mdio.schema.v1.units import AllUnits
from mdio.schema.v1.dataset import Dataset, DatasetMetadata
from mdio.schema.v1.variable import Variable, Coordinate, VariableMetadata
from mdio.schema.v1.template_factory import (
    make_named_dimension,
    make_coordinate,
    make_variable,
    make_dataset_metadata,
    make_dataset,
)


class _BuilderState(Enum):
    """States for the template builder."""
    INITIAL = auto()
    HAS_DIMENSIONS = auto()
    HAS_COORDINATES = auto()
    HAS_VARIABLES = auto()

class TemplateBuilder:
    """Builder for creating MDIO datasets with enforced build order:
    1. Must add dimensions first via add_dimension()
    2. Can optionally add coordinates via add_coordinate()
    3. Must add variables via add_variable()
    4. Must call build() to create the dataset
    """
    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.api_version = "1.0.0"  # TODO: Pull from package metadata
        self.created_on = datetime.now(timezone.utc)
        self.attributes = attributes
        self._dimensions: List[NamedDimension] = []
        self._coordinates: List[Coordinate] = []
        self._variables: List[Variable] = []
        self._state = _BuilderState.INITIAL
        self._unnamed_variable_counter = 0

    def add_dimension(self,
        name: str,
        size: int,
        long_name: str = None,
        data_type: ScalarType | StructuredType = ScalarType.INT32,
        metadata: Optional[List[AllUnits | UserAttributes]] | Dict[str, Any] = None) -> 'TemplateBuilder':
        """Add a dimension. This must be called at least once before adding coordinates or variables.
        
        Args:
            name: Name of the dimension
            size: Size of the dimension
            long_name: Optional long name for the dimension variable
            data_type: Data type for the dimension variable (defaults to INT32)
            metadata: Optional metadata for the dimension variable
        """
        # Create the dimension
        dimension = make_named_dimension(name, size)
        self._dimensions.append(dimension)
        
        # Create a variable for the dimension
        dim_var = make_variable(
            name=name,
            long_name=long_name,
            dimensions=[dimension],
            data_type=data_type,
            metadata=metadata
        )
        self._variables.append(dim_var)
        
        self._state = _BuilderState.HAS_DIMENSIONS
        return self

    def add_coordinate(self,
        name: str = "",
        *,
        long_name: str = None,
        dimensions: List[NamedDimension | str] = [],
        data_type: ScalarType | StructuredType = ScalarType.FLOAT32,
        metadata: Optional[List[AllUnits | UserAttributes]] | Dict[str, Any] = None) -> 'TemplateBuilder':
        """Add a coordinate after adding at least one dimension."""
        if self._state == _BuilderState.INITIAL:
            raise ValueError("Must add at least one dimension before adding coordinates")
        
        if name == "":
            name = f"coord_{len(self._coordinates)}"
        if dimensions == []:
            dimensions = self._dimensions
        if isinstance(metadata, dict):
            metadata = [metadata]
            
        # Convert string dimension names to NamedDimension objects
        dim_objects = []
        for dim in dimensions:
            if isinstance(dim, str):
                dim_obj = next((d for d in self._dimensions if d.name == dim), None)
                if dim_obj is None:
                    raise ValueError(f"Dimension '{dim}' not found")
                dim_objects.append(dim_obj)
            else:
                dim_objects.append(dim)
            
        self._coordinates.append(make_coordinate(
            name=name,
            long_name=long_name,
            dimensions=dim_objects,
            data_type=data_type,
            metadata=metadata
        ))
        self._state = _BuilderState.HAS_COORDINATES
        return self

    def add_variable(self,
        name: str = "",
        *,
        long_name: str = None,
        dimensions: List[NamedDimension | str] = [],
        data_type: ScalarType | StructuredType = ScalarType.FLOAT32,
        compressor: Blosc | ZFP | None = None,
        coordinates: Optional[List[Coordinate | str]] = None,
        metadata: Optional[VariableMetadata] = None) -> 'TemplateBuilder':
        """Add a variable after adding at least one dimension."""
        if self._state == _BuilderState.INITIAL:
            raise ValueError("Must add at least one dimension before adding variables")
            
        if name == "":
            name = f"var_{self._unnamed_variable_counter}"
            self._unnamed_variable_counter += 1
        if dimensions == []:
            dimensions = self._dimensions
            
        # Convert string dimension names to NamedDimension objects
        dim_objects = []
        for dim in dimensions:
            if isinstance(dim, str):
                dim_obj = next((d for d in self._dimensions if d.name == dim), None)
                if dim_obj is None:
                    raise ValueError(f"Dimension '{dim}' not found")
                dim_objects.append(dim_obj)
            else:
                dim_objects.append(dim)
            
        self._variables.append(make_variable(
            name=name,
            long_name=long_name,
            dimensions=dim_objects,
            data_type=data_type,
            compressor=compressor,
            coordinates=coordinates,
            metadata=metadata
        ))
        self._state = _BuilderState.HAS_VARIABLES
        return self

    def build(self) -> Dataset:
        """Build the final dataset."""
        if self._state == _BuilderState.INITIAL:
            raise ValueError("Must add at least one dimension before building")
            
        metadata = make_dataset_metadata(
            self.name,
            self.api_version,
            self.created_on,
            self.attributes
        )
        
        # Add coordinates as variables to the dataset
        # We make a copy so that coordinates are not duplicated if the builder is reused
        all_variables = self._variables.copy()
        for coord in self._coordinates:
            # Convert coordinate to variable
            coord_var = make_variable(
                name=coord.name,
                long_name=coord.long_name,
                dimensions=coord.dimensions,
                data_type=coord.data_type,
                metadata=coord.metadata
            )
            all_variables.append(coord_var)
            
        return make_dataset(all_variables, metadata)