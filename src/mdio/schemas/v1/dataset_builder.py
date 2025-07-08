"""Builder pattern implementation for MDIO v1 schema models."""

from datetime import UTC
from datetime import datetime
from enum import Enum
from enum import auto
from typing import Any
from typing import TypeAlias

from pydantic import BaseModel
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding  # noqa: F401

from mdio.schemas.compressors import ZFP
from mdio.schemas.compressors import Blosc
from mdio.schemas.dimension import NamedDimension
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset import DatasetInfo
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.variable import Coordinate
from mdio.schemas.v1.variable import Variable

AnyMetadataList: TypeAlias = list[AllUnits |
                                  UserAttributes |
                                  ChunkGridMetadata |
                                  StatisticsMetadata |
                                  DatasetInfo]
CoordinateMetadataList: TypeAlias = list[AllUnits | UserAttributes]
VariableMetadataList: TypeAlias = list[AllUnits |
                                       UserAttributes |
                                       ChunkGridMetadata |
                                       StatisticsMetadata]
DatasetMetadataList: TypeAlias = list[DatasetInfo | UserAttributes]


class _BuilderState(Enum):
    """States for the template builder."""

    INITIAL = auto()
    HAS_DIMENSIONS = auto()
    HAS_COORDINATES = auto()
    HAS_VARIABLES = auto()


def _get_named_dimension(
    dimensions: list[NamedDimension], name: str, size: int | None = None
) -> NamedDimension | None:
    """Get a dimension by name and optional size from the list[NamedDimension]."""
    if dimensions is None:
        return False
    if not isinstance(name, str):
        msg = f"Expected str, got {type(name).__name__}"
        raise TypeError(msg)

    nd = next((dim for dim in dimensions if dim.name == name), None)
    if nd is None:
        return None
    if size is not None and nd.size != size:
        msg = f"Dimension {name!r} found but size {nd.size} does not match expected size {size}"
        raise ValueError(msg)
    return nd


def _to_dictionary(val: BaseModel | dict[str, Any] | AnyMetadataList) -> dict[str, Any]:
    """Convert a dictionary, list or pydantic BaseModel to a dictionary."""
    if val is None:
        return None
    if isinstance(val, BaseModel):
        return val.model_dump(mode="json", by_alias=True)
    if isinstance(val, dict):
        return val
    if isinstance(val, list):
        metadata_dict = {}
        for md in val:
            if md is None:
                continue
            metadata_dict.update(_to_dictionary(md))
        return metadata_dict
    msg = f"Expected BaseModel, dict or list, got {type(val).__name__}"
    raise TypeError(msg)


class MDIODatasetBuilder:
    """Builder for creating MDIO datasets with enforced build order.

    This builder implements the builder pattern to create MDIO datasets with a v1 schema.
    It enforces a specific build order to ensure valid dataset construction:
    1. Must add dimensions first via add_dimension()
    2. Can optionally add coordinates via add_coordinate()
    3. Must add variables via add_variable()
    4. Must call build() to create the dataset.
    """

    def __init__(self, name: str, attributes: UserAttributes | None = None):

        info = DatasetInfo(
            name=name,
            api_version="1.0.0",
            created_on=datetime.now(UTC)
        )
        # TODO(BrianMichell, #0): Pull from package metadata
        self._info = info
        self._attributes = attributes
        self._dimensions: list[NamedDimension] = []
        self._coordinates: list[Coordinate] = []
        self._variables: list[Variable] = []
        self._state = _BuilderState.INITIAL
        self._unnamed_variable_counter = 0

    def add_dimension(  # noqa: PLR0913
        self,
        name: str,
        size: int
    ) -> "MDIODatasetBuilder":
        """Add a dimension.

        This function be called at least once before adding coordinates or variables.

        Args:
            name: Name of the dimension
            size: Size of the dimension

        Returns:
            self: Returns self for method chaining
        """
        if not name:
            msg = "'name' must be a non-empty string"
            raise ValueError(msg)

        # Validate that the dimension is not already defined
        old_var = next((e for e in self._dimensions if e.name == name), None)
        if old_var is not None:
            msg = "Adding dimension with the same name twice is not allowed"
            raise ValueError(msg)

        dim = NamedDimension(name=name, size=size)
        self._dimensions.append(dim)
        self._state = _BuilderState.HAS_DIMENSIONS
        return self

    def _get_coordinate(
        self,
        coordinates: list[Coordinate] | list[str],
        name: str, size: int | None = None
    ) -> Coordinate | None:
        """Get a coordinate by name from the list[Coordinate] | list[str]."""
        if coordinates is None:
            return None

        for c in coordinates:
            if isinstance(c, str) and c == name:
                # The coordinate is stored by name (str). 
                # Find it in the builder global list and return it.
                cc = next((v for v in self._coordinates if v.name == name), None)
                if cc is None:
                    msg = f"Pre-existing coordinate named {name!r} is not found"
                    raise ValueError(msg)
                return cc
            if isinstance(c, Coordinate) and c.name == name:
                # The coordinate is stored as an embedded Coordinate object.
                # Return it.
                return c

        return None

    def add_coordinate(  # noqa: PLR0913
        self,
        name: str,
        *,
        long_name: str = None,
        dimensions: list[str],
        data_type: ScalarType = ScalarType.FLOAT32,
        compressor: Blosc | ZFP | None = None,
        metadata_info: CoordinateMetadataList | None = None,
    ) -> "MDIODatasetBuilder":
        """Add a coordinate after adding at least one dimension.

        This function must be called after all required dimensions are added via add_dimension().
        This call will create a coordinate variable.

        Args:
            name: Name of the coordinate
            long_name: Optional long name for the coordinate
            dimensions: List of dimension names that the coordinate is associated with
            data_type: Data type for the coordinate (defaults to FLOAT32)
            metadata_info: Optional metadata information for the coordinate

        Returns:
            self: Returns self for method chaining
        """
        if self._state == _BuilderState.INITIAL:
            msg = "Must add at least one dimension before adding coordinates"
            raise ValueError(msg)
        if not name:
            msg = "'name' must be a non-empty string"
            raise ValueError(msg)
        if dimensions is None or not dimensions:
            msg = "'dimensions' must be a non-empty list"
            raise ValueError(msg)
        old_var = next((e for e in self._coordinates if e.name == name), None)
        # Validate that the coordinate is not already defined
        if old_var is not None:
            msg = "Adding coordinate with the same name twice is not allowed"
            raise ValueError(msg)

        # Validate that all referenced dimensions are already defined
        named_dimensions = []
        for dim_name in dimensions:
            nd = _get_named_dimension(self._dimensions, dim_name)
            if nd is None:
                msg = f"Pre-existing dimension named {dim_name!r} is not found"
                raise ValueError(msg)
            named_dimensions.append(nd)    

        meta_dict = _to_dictionary(metadata_info)
        coord = Coordinate(
            name=name,
            longName=long_name,
            dimensions=named_dimensions,
            compressor=compressor,
            dataType=data_type,
            metadata=meta_dict
        )
        self._coordinates.append(coord)

        # Add a coordinate variable to the dataset
        self.add_variable(
            name=coord.name,
            long_name=f"'{coord.name}' coordinate variable",
            dimensions=dimensions,  # dimension names (list[str])
            data_type=coord.data_type,
            compressor=compressor,
            coordinates=[name],  # Use the coordinate name as a reference
            metadata_info=coord.metadata
        )

        self._state = _BuilderState.HAS_COORDINATES
        return self

    def add_dimension_coordinate(
        self,
        dimension_name: str,
        *,
        data_type: ScalarType,
        compressor: Blosc | ZFP | None = None,
        metadata_info: VariableMetadataList | None = None,
    ) -> "MDIODatasetBuilder":
        """Add a dimension coordinate variable for a pre-existing dimension.
        This is a convenience method to create a coordinate variable
        that represents sampling along a dimension.
        """
        self.add_coordinate(dimension_name,
                            long_name=dimension_name,
                            dimensions=[dimension_name],
                            data_type=data_type,
                            compressor=compressor,
                            metadata_info=_to_dictionary(metadata_info))

        return self

    def add_variable(  # noqa: PLR0913
        self,
        name: str,
        *,
        long_name: str = None,
        dimensions: list[str],
        data_type: ScalarType | StructuredType,
        compressor: Blosc | ZFP | None = None,
        coordinates: list[str] | None = None,
        metadata_info: VariableMetadataList | None = None,
    ) -> "MDIODatasetBuilder":
        """Add a variable after adding at least one dimension and, optionally, coordinate.

        This function must be called after all required dimensions are added via add_dimension()
        This function must be called after all required coordinates are added via add_coordinate().

        If this function is called with a single dimension name that matches the variable name,
        it will create a dimension variable. Dimension variables are special variables that
        represent sampling along a dimension.

        Args:
            name: Name of the variable
            long_name: Optional long name for the variable
            dimensions: List of dimension names that the variable is associated with
            data_type: Data type for the variable (defaults to FLOAT32)
            compressor: Compressor used for the variable (defaults to None)
            coordinates: List of coordinate names that the variable is associated with
                         (defaults to None, meaning no coordinates)
            metadata_info: Optional metadata information for the variable

        Returns:
            self: Returns self for method chaining.
        """
        if self._state == _BuilderState.INITIAL:
            msg = "Must add at least one dimension before adding variables"
            raise ValueError(msg)
        if not name:
            msg = "'name' must be a non-empty string"
            raise ValueError(msg)
        if dimensions is None or not dimensions:
            msg = "'dimensions' must be a non-empty list"
            raise ValueError(msg)

        # Validate that the variable is not already defined
        old_var = next((e for e in self._variables if e.name == name), None)
        if old_var is not None:
            msg = "Adding variable with the same name twice is not allowed"
            raise ValueError(msg)

        # Validate that all referenced dimensions are already defined
        named_dimensions = []
        for dim_name in dimensions:
            nd = _get_named_dimension(self._dimensions, dim_name)
            if nd is None:
                msg = f"Pre-existing dimension named {dim_name!r} is not found"
                raise ValueError(msg)
            named_dimensions.append(nd)

        # Validate that all referenced coordinates are already defined
        if coordinates is not None:
            for coord in coordinates:
                if next((c for c in self._coordinates if c.name == coord), None) is None:
                    msg = f"Pre-existing coordinate named {coord!r} is not found"
                    raise ValueError(msg)

        meta_dict = _to_dictionary(metadata_info)
        var = Variable(
            name=name,
            long_name=long_name,
            dimensions=named_dimensions,
            data_type=data_type,
            compressor=compressor,
            coordinates=coordinates,
            metadata=meta_dict)
        self._variables.append(var)

        self._state = _BuilderState.HAS_VARIABLES
        return self

    def build(self) -> Dataset:
        """Build the final dataset.

        This function must be called after at least one dimension is added via add_dimension().
        It will create a Dataset object with all added dimensions, coordinates, and variables.

        Returns:
            Dataset: The built dataset with all added dimensions, coordinates, and variables.
        """
        if self._state == _BuilderState.INITIAL:
            msg = "Must add at least one dimension before building"
            raise ValueError(msg)

        var_meta_dict = _to_dictionary([self._info, self._attributes])
        return Dataset(variables=self._variables, metadata=var_meta_dict)
