"""Factory methods for MDIO v1 schema models."""

from datetime import datetime
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
    dimensions: List[Union[NamedDimension, str]],
    data_type: ScalarType,
    metadata: Optional[List[Union[AllUnits, UserAttributes]]] = None,
) -> Coordinate:
    """Create a Coordinate with the given name, dimensions, data_type, and metadata."""
    return Coordinate(
        name=name,
        dimensions=dimensions,
        data_type=data_type,
        metadata=metadata,
    )


def make_variable(
    name: str,
    dimensions: List[Union[NamedDimension, str]],
    data_type: Union[ScalarType, StructuredType],
    compressor: Union[Blosc, ZFP, None],
    coordinates: Optional[List[Union[Coordinate, str]]] = None,
    metadata: Optional[VariableMetadata] = None,
) -> Variable:
    """Create a Variable with the given name, dimensions, data_type, compressor, coordinates, and metadata."""
    return Variable(
        name=name,
        dimensions=dimensions,
        data_type=data_type,
        compressor=compressor,
        coordinates=coordinates,
        metadata=metadata,
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