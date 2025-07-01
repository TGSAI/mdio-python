"""Builder pattern implementation for MDIO v1 schema models."""

from datetime import UTC
from datetime import datetime
from enum import Enum
from enum import auto
from typing import Any
from typing import TypeAlias

import xarray as xr

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
from mdio.schemas.v1.dataset import Dataset, DatasetInfo
from mdio.schemas.v1.dataset import DatasetMetadata

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

        added_dims = self._add_dimensions_if_needed([NamedDimension(name=name, size=size)])
        if added_dims:
            meta_dict = _to_dictionary(var_metadata_info)
            # Create a variable for the dimension
            dim_var = Variable(
                name=name,
                longName=var_long_name,
                # IMPORTANT: we always use NamedDimension here, not the dimension name
                # Since the Dataset does not have a dimension list, we need to preserve NamedDimension
                # somewhere. Namely, in the variable created for the dimension
                dimensions=added_dims,
                dataType=var_data_type,
                compressor=None,
                coordinates=None,
                metadata=meta_dict,
            )
            self._variables.append(dim_var)

        self._state = _BuilderState.HAS_DIMENSIONS
        return self

    def add_coordinate(  # noqa: PLR0913
        self,
        name: str,
        *,
        long_name: str = None,
        #TODO Only allow adding dimensions by name, not by NamedDimension object
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
        meta_dict = _to_dictionary(metadata_info)
        coord = Coordinate(
            name=name,
            longName=long_name,
            # We ass names: sts, not list[NamedDimension | str]
            dimensions=dim_names,
            dataType=data_type,
            metadata=meta_dict
        )
        self._coordinates.append(coord)

        # Add coordinate as variables to the dataset
        var_meta_dict = _to_dictionary(coord.metadata)
        coord_var = Variable(
            name=coord.name,
            longName=coord.long_name,
            dimensions=coord.dimensions,
            dataType=coord.data_type,
            compressor=None,
            # IMPORTANT: we always use Coordinate here, not the coordinate name
            # Since the Dataset does not have a coordinate list, we need to preserve Coordinate
            # somewhere. Namely, in the variable created for the coordinate
            coordinates=[coord],
            metadata=var_meta_dict
        )
        self._variables.append(coord_var)

        self._state = _BuilderState.HAS_COORDINATES
        return self

    def add_variable(  # noqa: PLR0913
        self,
        name: str,
        *,
        long_name: str = None,
        #TODO Only allow adding dimensions by name, not by NamedDimension object
        dimensions: list[NamedDimension | str],
        data_type: ScalarType | StructuredType = ScalarType.FLOAT32,
        compressor: Blosc | ZFP | None = None,
        #TODO Only allow adding coordinates by name, not by Coordinate object
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
        meta_dict = _to_dictionary(metadata_info)
        self._variables.append(
            Variable(
                name=name,
                long_name=long_name,
                dimensions=dim_names,
                data_type=data_type,
                compressor=compressor,
                coordinates=coordinates,
                metadata=meta_dict,
            )
        )
        self._state = _BuilderState.HAS_VARIABLES
        return self

    def build(self) -> Dataset:
        """Build the final dataset."""
        if self._state == _BuilderState.INITIAL:
            msg = "Must add at least one dimension before building"
            raise ValueError(msg)

        var_meta_dict = _to_dictionary([self._info, self._attributes])
        dataset = Dataset(variables=self._variables, metadata=var_meta_dict)

        return dataset

    # def to_mdio(
    #     self,
    #     store: str,
    #     mode: str = "w",
    #     compute: bool = False,
    #     **kwargs: Mapping[str, str | int | float | bool],
    # ) -> Dataset:
    #     """Write the dataset to a Zarr store and return the constructed mdio.Dataset.

    #     This function constructs an mdio.Dataset from the MDIO dataset and writes its metadata
    #     to a Zarr store. The actual data is not written, only the metadata structure is created.
    #     """
    #     return write_mdio_metadata(self.build(), store, mode, compute, **kwargs)

# def write_mdio_metadata(
#     mdio_ds: Dataset,
#     store: str,
#     mode: str = "w",
#     compute: bool = False,
#     **kwargs: Mapping[str, str | int | float | bool],
# ) -> mdio.Dataset:
#     """Write MDIO metadata to a Zarr store and return the constructed mdio.Dataset.

#     This function constructs an mdio.Dataset from the MDIO dataset and writes its metadata
#     to a Zarr store. The actual data is not written, only the metadata structure is created.

#     Args:
#         mdio_ds: The MDIO dataset to serialize
#         store: Path to the Zarr or .mdio store
#         mode: Write mode to pass to to_mdio(), e.g. 'w' or 'a'
#         compute: Whether to compute (write) array chunks (True) or only metadata (False)
#         **kwargs: Additional arguments to pass to to_mdio()

#     Returns:
#         The constructed xarray Dataset with MDIO extensions
#     """
#     ds = _construct_mdio_dataset(mdio_ds)

#     def _generate_encodings() -> dict:
#         """Generate encodings for each variable in the MDIO dataset.

#         Returns:
#             Dictionary mapping variable names to their encoding configurations.
#         """
#         # TODO(Anybody, #10274): Re-enable chunk_key_encoding when supported by xarray
#         # dimension_separator_encoding = V2ChunkKeyEncoding(separator="/").to_dict()

#         # Collect dimension sizes (same approach as _construct_mdio_dataset)
#         dims: dict[str, int] = {}
#         for var in mdio_ds.variables:
#             for d in var.dimensions:
#                 if isinstance(d, NamedDimension):
#                     dims[d.name] = d.size

#         global_encodings = {}
#         for var in mdio_ds.variables:
#             fill_value = 0
#             if isinstance(var.data_type, StructuredType):
#                 continue
#             chunks = None
#             if var.metadata is not None and var.metadata.chunk_grid is not None:
#                 chunks = var.metadata.chunk_grid.configuration.chunk_shape
#             else:
#                 # When no chunk_grid is provided, set chunks to shape to avoid chunking
#                 dim_names = [d.name if isinstance(d, NamedDimension) else d for d in var.dimensions]
#                 chunks = tuple(dims[name] for name in dim_names)
#             global_encodings[var.name] = {
#                 "chunks": chunks,
#                 # TODO(Anybody, #10274): Re-enable chunk_key_encoding when supported by xarray
#                 # "chunk_key_encoding": dimension_separator_encoding,
#                 "_FillValue": fill_value,
#                 "dtype": var.data_type,
#                 "compressors": _convert_compressor(var.compressor),
#             }
#         return global_encodings

#     ds.to_mdio(
#         store,
#         mode=mode,
#         zarr_format=2,
#         consolidated=True,
#         safe_chunks=False,
#         compute=compute,
#         encoding=_generate_encodings(),
#         **kwargs,
#     )
#     return ds
