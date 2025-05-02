"""Factory methods for MDIO v1 schema models."""

from datetime import datetime, timezone
from typing import Any, Optional, Union, List, Dict

from pydantic import AwareDatetime

from mdio.schema.dimension import NamedDimension
from mdio.schema.compressors import Blosc, ZFP
from mdio.schema.dtype import ScalarType, StructuredType
from mdio.schema.metadata import UserAttributes
from mdio.schema.v1.units import AllUnits
from mdio.schema.v1.dataset import Dataset, DatasetMetadata
from mdio.schema.v1.variable import Variable, Coordinate, VariableMetadata


def make_named_dimension(name: str, size: int) -> NamedDimension:
    """Create a NamedDimension with the given name and size."""
    return NamedDimension(name=name, size=size)


def make_coordinate(
    name: str,
    dimensions: List[NamedDimension | str],
    data_type: ScalarType | StructuredType,
    long_name: str = None,
    metadata: Optional[List[AllUnits | UserAttributes]] = None,
) -> Coordinate:
    """Create a Coordinate with the given name, dimensions, data_type, and metadata."""
    return Coordinate(
        name=name,
        long_name=long_name,
        dimensions=dimensions,
        data_type=data_type,
        metadata=metadata,
    )


def make_variable(
    name: str,
    dimensions: List[NamedDimension | str],
    data_type: ScalarType | StructuredType,
    long_name: str = None,
    compressor: Blosc | ZFP | None = None,
    coordinates: Optional[List[Coordinate | str]] = None,
    metadata: Optional[List[AllUnits | UserAttributes] | Dict[str, Any] | VariableMetadata] = None,
) -> Variable:
    """Create a Variable with the given name, dimensions, data_type, compressor, coordinates, and metadata."""
    # Convert metadata to VariableMetadata if needed
    var_metadata = None
    if metadata:
        if isinstance(metadata, list):
            # Convert list of metadata to dict
            metadata_dict = {}
            for md in metadata:
                if isinstance(md, AllUnits):
                    # For units_v1, if it's a single element, use it directly
                    if isinstance(md.units_v1, list) and len(md.units_v1) == 1:
                        metadata_dict["units_v1"] = md.units_v1[0]
                    else:
                        metadata_dict["units_v1"] = md.units_v1
                elif isinstance(md, UserAttributes):
                    # For attributes, if it's a single element, use it directly
                    attrs = md.model_dump(by_alias=True)
                    if isinstance(attrs, list) and len(attrs) == 1:
                        metadata_dict["attributes"] = attrs[0]
                    else:
                        metadata_dict["attributes"] = attrs
            var_metadata = VariableMetadata(**metadata_dict)
        elif isinstance(metadata, dict):
            # Convert camelCase keys to snake_case for VariableMetadata
            converted_dict = {}
            for key, value in metadata.items():
                if key == "unitsV1":
                    # For units_v1, if it's a single element array, use the element directly
                    if isinstance(value, list) and len(value) == 1:
                        converted_dict["units_v1"] = value[0]
                    else:
                        converted_dict["units_v1"] = value
                else:
                    converted_dict[key] = value
            var_metadata = VariableMetadata(**converted_dict)
        elif isinstance(metadata, VariableMetadata):
            var_metadata = metadata
        else:
            raise TypeError(f"Unsupported metadata type: {type(metadata)}")

    # Create the variable with all attributes explicitly set
    return Variable(
        name=name,
        long_name=long_name,
        dimensions=dimensions,
        data_type=data_type,
        compressor=compressor,
        coordinates=coordinates,
        metadata=var_metadata,
    )


def make_dataset_metadata(
    name: str,
    api_version: str,
    created_on: AwareDatetime,
    attributes: Optional[Dict[str, Any]] = None,
) -> DatasetMetadata:
    """Create a DatasetMetadata with name, api_version, created_on, and optional attributes."""
    return DatasetMetadata(
        name=name,
        api_version=api_version,
        created_on=created_on,
        attributes=attributes,
    )


def make_dataset(
    variables: List[Variable],
    metadata: DatasetMetadata,
) -> Dataset:
    """Create a Dataset with the given variables and metadata."""
    return Dataset(
        variables=variables,
        metadata=metadata,
    )


class AbstractTemplateFactory:

    def __init__(self, name: str): 
        self.name = name
        self.api_version = "1.0.0"  # TODO: Pull from package metadata
        self.created_on = datetime.now(timezone.utc)


    def AddDimension(self, name: str, size: int) -> 'AbstractTemplateFactory':
        """Add a dimension to the factory."""
        self.dimensions.append(make_named_dimension(name, size))
        return self


    def AddCoordinate(self, 
        name: str = "",
        dimensions: List[NamedDimension | str] = [],
        data_type: ScalarType | StructuredType = ScalarType.FLOAT32,
        metadata: Optional[List[AllUnits | UserAttributes]] = None) -> 'AbstractTemplateFactory':
        """Add a coordinate to the factory."""
        if name == "":
            name = f"coord_{len(self.coordinates)}"
        if dimensions == []:
            dimensions = self.dimensions
        self.coordinates.append(make_coordinate(name, dimensions, data_type, metadata))
        return self

    def AddVariable(self, name: str = "",
            dimensions: List[NamedDimension | str] = [],
            data_type: ScalarType | StructuredType = ScalarType.FLOAT32,
            compressor: Blosc | ZFP | None = None, 
            coordinates: Optional[List[Coordinate | str]] = None, 
            metadata: Optional[VariableMetadata] = None) -> 'AbstractTemplateFactory':
        """Add a variable to the factory."""
        if name == "":
            name = f"var_{len(self.variables)}"
        if dimensions == []:
            dimensions = self.dimensions
        self.variables.append(make_variable(name, dimensions, data_type, compressor, coordinates, metadata))
        return self

    def _compose_metadata(self) -> DatasetMetadata:
        """Compose the DatasetMetadata with the given name, api_version, and created_on."""
        return make_dataset_metadata(self.name, self.api_version, self.created_on)


    def _compose_variables(self) -> List[Variable]:
        """Compose the Variables with the given name, dimensions, data_type, compressor, coordinates, and metadata."""
        return [
            make_variable(self.name, self.dimensions, self.data_type, self.compressor, self.coordinates, self.metadata)
        ]


    def make_dataset(self, variables: List[Variable]) -> Dataset:
        """Create a Dataset with the given variables and metadata."""
        return Dataset(variables=variables, metadata=self._compose_metadata())
