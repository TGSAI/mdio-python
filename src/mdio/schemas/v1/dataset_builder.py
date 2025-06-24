"""Builder pattern implementation for MDIO v1 schema models."""

from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from enum import Enum
from enum import auto
from typing import Any

from pydantic import BaseModel
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding  # noqa: F401

from mdio.schemas.compressors import ZFP
from mdio.schemas.compressors import Blosc
from mdio.schemas.dimension import NamedDimension
from mdio.schemas.dtype import ScalarType, StructuredType
from mdio.schemas.metadata import ChunkGridMetadata, UserAttributes
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.stats import StatisticsMetadata, SummaryStatistics
from mdio.schemas.v1.units import AllUnitModel, AllUnits
from mdio.schemas.v1.variable import Coordinate, Variable, VariableMetadata

# TODO: Why do we allow default names?
# TODO: Instead of trying to track the state, should we just return a wrapper class with permitted methods?
# TODO: refactor to make inner class
class _BuilderState(Enum):
    """States for the template builder."""

    INITIAL = auto()
    HAS_DIMENSIONS = auto()
    HAS_COORDINATES = auto()
    HAS_VARIABLES = auto()

def contains_dimension(dimensions: list[NamedDimension], name_or_NamedDimension: str | NamedDimension) -> bool:
    """Check if a dimension with the given name exists in the list."""
    if isinstance(name_or_NamedDimension, str):
        name = name_or_NamedDimension
        return get_dimension(dimensions, name) is not None
    elif isinstance(name_or_NamedDimension, NamedDimension):
        named_dimension = name_or_NamedDimension
        return get_dimension(dimensions, named_dimension.name, named_dimension.size) is not None
    else:
        msg = f"Expected str or NamedDimension, got {type(name_or_NamedDimension).__name__}"
        raise TypeError(msg)

def get_dimension(dimensions: list[NamedDimension], name: str, size: int | None = None) -> NamedDimension | None:
    """Get a dimension by name from the list."""
    if dimensions is None:
        return False
    if not isinstance(name, str):
        raise TypeError(f"Expected str, got {type(name).__name__}")

    nd = next((dim for dim in dimensions if dim.name == name), None)
    if nd is None:
        return None
    if size is not None and nd.size != size:
        msg = f"Dimension {name!r} found but size {nd.size} does not match expected size {size}"
        raise ValueError(msg)
    return nd

def to_dictionary(val: BaseModel) -> dict[str, Any]:
    """Convert a pydantic BaseModel to a dictionary."""
    if not isinstance(val, BaseModel):
        raise TypeError(f"Expected BaseModel, got {type(val).__name__}")
    return val.model_dump(mode="json", by_alias=True) 

class MDIODatasetBuilder:
    """Builder for creating MDIO datasets with enforced build order.

    This builder implements the builder pattern to create MDIO datasets with a v1 schema.
    It enforces a specific build order to ensure valid dataset construction:
    1. Must add dimensions first via add_dimension()
    2. Can optionally add coordinates via add_coordinate()
    3. Must add variables via add_variable()
    4. Must call build() to create the dataset.
    """

    def __init__(self, name: str, attributes: dict[str, Any] | None = None):
        self.name = name
        self.api_version = "1.0.0"  # TODO(BrianMichell, #0): Pull from package metadata
        self.created_on = datetime.now(UTC)
        self.attributes = attributes
        self._dimensions: list[NamedDimension] = []
        self._coordinates: list[Coordinate] = []
        self._variables: list[Variable] = []
        self._state = _BuilderState.INITIAL
        self._unnamed_variable_counter = 0


    def _add_named_dimensions(self, dimensions: list[NamedDimension | str] | None) -> list[NamedDimension]:
        if dimensions is None:
            return []

        added_dims = []
        for dim in dimensions:
            if isinstance(dim, str):
                if not contains_dimension(self._dimensions, dim):
                    raise ValueError(f"Dimension named {dim!r} is not found")
            else:
                if not isinstance(dim, NamedDimension):
                    raise TypeError(f"Expected NamedDimension or str, got {type(dim).__name__}")
                if  contains_dimension(self._dimensions, dim):
                    continue
                self._dimensions.append(dim)
                added_dims.append(dim)
        return added_dims


    def _make_VariableMetadata_from_list(metadata: list[AllUnits | UserAttributes]) -> Any:
        metadata_dict = {}
        for md in metadata:
            # NOTE: the pydentic attribute names are different from the v1 schema attributes names
            #       'statsV1' <-> 'stats_v1', 'unitsV1' <-> 'units_v1', 'chunkGrid' <-> 'chunk_grid'
            if isinstance(md, AllUnits):
                val = md.units_v1
                metadata_dict["unitsV1"] = to_dictionary(val)
            elif isinstance(md, UserAttributes):
                # NOTE: md.attributes is not pydantic type, but a dictionary
                metadata_dict["attributes"] = to_dictionary(md)["attributes"]
            elif isinstance(md, ChunkGridMetadata):
                val = md.chunk_grid
                metadata_dict["chunkGrid"] = to_dictionary(val)
            elif isinstance(md, StatisticsMetadata):
                val = md.stats_v1
                metadata_dict["statsV1"] = to_dictionary(val)
            else:
                raise TypeError(f"Unsupported metadata type: {type(md)}")
        return VariableMetadata(**metadata_dict)


    def _make_VariableMetadata_from_dict(metadata: dict[str, Any]) -> type[BaseModel]:
        converted_dict = {}
        for key, value in metadata.items():
            if key == "unitsV1" or key == "statsV1" or key == "chunkGrid" or key == "attributes":
                # TODO: Should we validate the structure of the value passed in?
                if not isinstance(value, dict):
                    raise TypeError(f"Invalid value for key '{key}': {value!r}. Expected a dictionary.")
            else:
                raise TypeError(f"Unsupported metadata key: '{key}'. Expected 'unitsV1', 'attributes', 'chunkGrid', or 'statsV1.")
            converted_dict[key] = value
        return VariableMetadata(**converted_dict)


    def _make_VariableMetadata(metadata: list[AllUnits | UserAttributes] | dict[str, Any] | None = None) -> Any | None:
        if metadata is None:
            return None
        
        if isinstance(metadata, list):
            return MDIODatasetBuilder._make_VariableMetadata_from_list(metadata)
        
        if isinstance(metadata, dict):
            return MDIODatasetBuilder._make_VariableMetadata_from_dict(metadata)

        raise TypeError(f"Unsupported metadata type: {type(metadata)}")


    def add_dimension(  # noqa: PLR0913
        self,
        name: str,
        size: int,
        long_name: str = None,
        data_type: ScalarType | StructuredType = ScalarType.INT32,
        metadata: list[AllUnits | UserAttributes] | None | dict[str, Any] = None,
    ) -> "MDIODatasetBuilder":
        """Add a dimension.

        This must be called at least once before adding coordinates or variables.

        Args:
            name: Name of the dimension
            size: Size of the dimension
            long_name: Optional long name for the dimension variable
            data_type: Data type for the dimension variable (defaults to INT32)
            metadata: Optional metadata for the dimension variable

        Returns:
            self: Returns self for method chaining
        """

        added_dims = self._add_named_dimensions([NamedDimension(name=name, size=size)])
        if added_dims:
            # Create a variable for the dimension
            dim_var = Variable(
                name=name,
                longName=long_name,
                dimensions=added_dims,
                dataType=data_type,
                compressor=None,
                coordinates=None,
                metadata=MDIODatasetBuilder._make_VariableMetadata(metadata)
            )
            self._variables.append(dim_var)

        self._state = _BuilderState.HAS_DIMENSIONS
        return self
