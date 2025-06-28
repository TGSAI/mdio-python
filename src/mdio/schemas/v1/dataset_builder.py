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
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.variable import Coordinate
from mdio.schemas.v1.variable import CoordinateMetadata
from mdio.schemas.v1.variable import Variable
from mdio.schemas.v1.variable import VariableMetadata

CoordinateMetadataList: TypeAlias = list[AllUnits |
                                         UserAttributes]
VariableMetadataList: TypeAlias = list[AllUnits |
                                       UserAttributes |
                                       ChunkGridMetadata |
                                       StatisticsMetadata]


class _BuilderState(Enum):
    """States for the template builder."""

    INITIAL = auto()
    HAS_DIMENSIONS = auto()
    HAS_COORDINATES = auto()
    HAS_VARIABLES = auto()


def contains_dimension(
    dimensions: list[NamedDimension], name_or_dimension: str | NamedDimension
) -> bool:
    """Check if a dimension with the given name exists in the list."""
    if isinstance(name_or_dimension, str):
        name = name_or_dimension
        return get_dimension(dimensions, name) is not None
    if isinstance(name_or_dimension, NamedDimension):
        dimension = name_or_dimension
        return get_dimension(dimensions, dimension.name, dimension.size) is not None
    msg = f"Expected str or NamedDimension, got {type(name_or_dimension).__name__}"
    raise TypeError(msg)


def get_dimension(
    dimensions: list[NamedDimension], name: str, size: int | None = None
) -> NamedDimension | None:
    """Get a dimension by name from the list."""
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


def get_dimension_names(dimensions: list[NamedDimension | str]) -> list[str]:
    """Get a dimension by name from the list."""
    names = []
    if dimensions is None:
        return names
    for dim in dimensions:
        if isinstance(dim, NamedDimension):
            names.append(dim.name)
        elif isinstance(dim, str):
            names.append(dim)
    return names


def _to_dictionary(val: BaseModel) -> dict[str, Any]:
    """Convert a pydantic BaseModel to a dictionary."""
    if not isinstance(val, BaseModel):
        msg = f"Expected BaseModel, got {type(val).__name__}"
        raise TypeError(msg)
    return val.model_dump(mode="json", by_alias=True)


def _make_coordinate_metadata(metadata: CoordinateMetadataList | None) -> CoordinateMetadata | None:
    if metadata is None or not metadata:
        return None

    metadata_dict = {}
    for md in metadata:
        # NOTE: the pydantic attribute names are different from the v1 schema attributes names
        #       'unitsV1' <-> 'units_v1'
        if isinstance(md, AllUnits):
            val = md.units_v1
            metadata_dict["unitsV1"] = _to_dictionary(val)
        elif isinstance(md, UserAttributes):
            # NOTE: md.attributes is not pydantic type, but a dictionary
            metadata_dict["attributes"] = _to_dictionary(md)["attributes"]
        else:
            msg = f"Unsupported metadata type: {type(md)}"
            raise TypeError(msg)
    return CoordinateMetadata(**metadata_dict)


def _make_variable_metadata(metadata: VariableMetadataList | None) -> VariableMetadata | None:
    if metadata is None or not metadata:
        return None

    metadata_dict = {}
    for md in metadata:
        # NOTE: the pydantic attribute names are different from the v1 schema attributes names
        #       'statsV1' <-> 'stats_v1', 'unitsV1' <-> 'units_v1', 'chunkGrid' <-> 'chunk_grid'
        if isinstance(md, AllUnits):
            val = md.units_v1
            metadata_dict["unitsV1"] = _to_dictionary(val)
        elif isinstance(md, UserAttributes):
            # NOTE: md.attributes is not pydantic type, but a dictionary
            metadata_dict["attributes"] = _to_dictionary(md)["attributes"]
        elif isinstance(md, ChunkGridMetadata):
            val = md.chunk_grid
            metadata_dict["chunkGrid"] = _to_dictionary(val)
        elif isinstance(md, StatisticsMetadata):
            val = md.stats_v1
            metadata_dict["statsV1"] = _to_dictionary(val)
        else:
            msg = f"Unsupported metadata type: {type(md)}"
            raise TypeError(msg)
    return VariableMetadata(**metadata_dict)


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
        # TODO(BrianMichell, #0): Pull from package metadata
        self.api_version = "1.0.0"
        self.created_on = datetime.now(UTC)
        self.attributes = attributes
        self._dimensions: list[NamedDimension] = []
        self._coordinates: list[Coordinate] = []
        self._variables: list[Variable] = []
        self._state = _BuilderState.INITIAL
        self._unnamed_variable_counter = 0

    def _add_dimensions_if_needed(
        self, dimensions: list[NamedDimension | str] | None
    ) -> list[NamedDimension]:
        if dimensions is None:
            return []

        added_dims = []
        for dim in dimensions:
            if isinstance(dim, str):
                if not contains_dimension(self._dimensions, dim):
                    msg = f"Pre-existing dimension named {dim!r} is not found"
                    raise ValueError(msg)
            else:
                if not isinstance(dim, NamedDimension):
                    msg = f"Expected NamedDimension or str, got {type(dim).__name__}"
                    raise TypeError(msg)
                if contains_dimension(self._dimensions, dim):
                    continue
                # Use value instead of a reference
                d = NamedDimension(name=dim.name, size=dim.size)
                self._dimensions.append(d)
                added_dims.append(d)
        return added_dims

    def add_dimension(  # noqa: PLR0913
        self,
        name: str,
        size: int,
        var_long_name: str = None,
        var_data_type: ScalarType | StructuredType = ScalarType.INT32,
        var_metadata_info: VariableMetadataList | None = None,
    ) -> "MDIODatasetBuilder":
        """Add a dimension.

        This must be called at least once before adding coordinates or variables.
        This call will create a variable, if one does not yet exists

        Args:
            name: Name of the dimension
            size: Size of the dimension
            var_long_name: Optional long name for the dimension variable
            var_data_type: Data type for the dimension variable (defaults to INT32)
            var_metadata_info: Optional metadata information for the dimension variable

        Returns:
            self: Returns self for method chaining
        """
        if not name:
            msg = "'name' must be a non-empty string"
            raise ValueError(msg)
        old_var = next((e for e in self._dimensions if e.name == name), None)
        if old_var is not None:
            msg = "Adding dimension with the same name twice is not allowed"
            raise ValueError(msg)

        added_dims = self._add_dimensions_if_needed(
            [NamedDimension(name=name, size=size)])
        if added_dims:
            # Create a variable for the dimension
            dim_var = Variable(
                name=name,
                longName=var_long_name,
                dimensions=added_dims,
                dataType=var_data_type,
                compressor=None,
                coordinates=None,
                metadata=_make_variable_metadata(var_metadata_info),
            )
            self._variables.append(dim_var)

        self._state = _BuilderState.HAS_DIMENSIONS
        return self

    def add_coordinate(  # noqa: PLR0913
        self,
        name: str,
        *,
        long_name: str = None,
        dimensions: list[NamedDimension | str],
        data_type: ScalarType | StructuredType = ScalarType.FLOAT32,
        metadata_info: CoordinateMetadataList | None = None,
    ) -> "MDIODatasetBuilder":
        """Add a coordinate after adding at least one dimension."""
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
        if old_var is not None:
            msg = "Adding coordinate with the same name twice is not allowed"
            raise ValueError(msg)

        self._add_dimensions_if_needed(dimensions)
        dim_names = get_dimension_names(dimensions)
        self._coordinates.append(
            Coordinate(
                name=name,
                longName=long_name,
                # We ass names: sts, not list[NamedDimension | str]
                dimensions=dim_names,
                dataType=data_type,
                metadata=_make_coordinate_metadata(metadata_info),
            )
        )
        self._state = _BuilderState.HAS_COORDINATES
        return self

    def add_variable(  # noqa: PLR0913
        self,
        name: str,
        *,
        long_name: str = None,
        dimensions: list[NamedDimension | str],
        data_type: ScalarType | StructuredType = ScalarType.FLOAT32,
        compressor: Blosc | ZFP | None = None,
        coordinates: list[Coordinate | str] | None = None,
        metadata_info: VariableMetadataList | None = None,
    ) -> "MDIODatasetBuilder":
        """Add a variable after adding at least one dimension."""
        if self._state == _BuilderState.INITIAL:
            msg = "Must add at least one dimension before adding variables"
            raise ValueError(msg)
        if not name:
            msg = "'name' must be a non-empty string"
            raise ValueError(msg)
        if dimensions is None or not dimensions:
            msg = "'dimensions' must be a non-empty list"
            raise ValueError(msg)
        old_var = next((e for e in self._variables if e.name == name), None)
        if old_var is not None:
            msg = "Adding variable with the same name twice is not allowed"
            raise ValueError(msg)

        self._add_dimensions_if_needed(dimensions)
        dim_names = get_dimension_names(dimensions)
        self._variables.append(
            Variable(
                name=name,
                long_name=long_name,
                dimensions=dim_names,
                data_type=data_type,
                compressor=compressor,
                coordinates=coordinates,
                metadata=_make_variable_metadata(metadata_info),
            )
        )
        self._state = _BuilderState.HAS_VARIABLES
        return self
