"""Template method pattern implementation for MDIO v1 dataset template."""

import copy
from abc import ABC
from abc import abstractmethod
from typing import Any

from mdio.builder.dataset_builder import MDIODatasetBuilder
from mdio.builder.schemas import compressors
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.v1.dataset import Dataset
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.builder.templates.types import SeismicDataDomain


class AbstractDatasetTemplate(ABC):
    """Abstract base class that defines the template method for Dataset building factory.

    The template method defines the skeleton of the data processing algorithm, while allowing subclasses
    to override specific steps.
    """

    def __init__(self, data_domain: SeismicDataDomain) -> None:
        self._data_domain = data_domain.lower()

        if self._data_domain not in ["depth", "time"]:
            msg = "domain must be 'depth' or 'time'"
            raise ValueError(msg)

        self._coord_dim_names = ()
        self._dim_names = ()
        self._coord_names = ()
        self._var_chunk_shape = ()

        self._builder: MDIODatasetBuilder | None = None
        self._dim_sizes = ()
        self._horizontal_coord_unit = None

    def build_dataset(
        self,
        name: str,
        sizes: tuple[int, ...],
        horizontal_coord_unit: LengthUnitModel,
        header_dtype: StructuredType = None,
    ) -> Dataset:
        """Template method that builds the dataset.

        Args:
            name: The name of the dataset.
            sizes: The sizes of the dimensions.
            horizontal_coord_unit: The units for the horizontal coordinates.
            header_dtype: Optional structured headers for the dataset.

        Returns:
            Dataset: The constructed dataset
        """
        self._dim_sizes = sizes
        self._horizontal_coord_unit = horizontal_coord_unit

        attributes = self._load_dataset_attributes() or {}
        attributes["defaultVariableName"] = self._default_variable_name
        self._builder = MDIODatasetBuilder(name=name, attributes=attributes)
        self._add_dimensions()
        self._add_coordinates()
        self._add_variables()
        self._add_trace_mask()
        if header_dtype:
            self._add_trace_headers(header_dtype)
        return self._builder.build()

    @property
    def name(self) -> str:
        """Returns the name of the template."""
        return self._name

    @property
    def default_variable_name(self) -> str:
        """Returns the name of the trace variable."""
        return self._default_variable_name

    @property
    def trace_domain(self) -> str:
        """Returns the name of the trace domain."""
        return self._data_domain

    @property
    def dimension_names(self) -> tuple[str, ...]:
        """Returns the names of the dimensions."""
        return copy.deepcopy(self._dim_names)

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        """Returns the names of the coordinates."""
        return copy.deepcopy(self._coord_names)

    @property
    def full_chunk_size(self) -> tuple[int, ...]:
        """Returns the chunk size for the variables."""
        return copy.deepcopy(self._var_chunk_shape)

    @property
    @abstractmethod
    def _name(self) -> str:
        """Abstract method to get the name of the template.

        Must be implemented by subclasses.

        Returns:
            The name of the template
        """

    @property
    def _default_variable_name(self) -> str:
        """Get the name of the data variable.

        A virtual method that can be overwritten by subclasses to return a
        custom data variable name.

        Returns:
            The name of the data variable
        """
        return "amplitude"

    @abstractmethod
    def _load_dataset_attributes(self) -> dict[str, Any]:
        """Abstract method to load dataset attributes.

        Must be implemented by subclasses.

        Returns:
            The dataset attributes as a dictionary
        """

    def _add_dimensions(self) -> None:
        """Add custom dimensions.

        A virtual method that can be overwritten by subclasses to add custom dimensions.
        Uses the class field 'builder' to add dimensions to the dataset.
        """
        for i in range(len(self._dim_names)):
            self._builder.add_dimension(self._dim_names[i], self._dim_sizes[i])

    def _add_coordinates(self) -> None:
        """Add custom coordinates.

        A virtual method that can be overwritten by subclasses to add custom coordinates.
        Uses the class field 'builder' to add coordinates to the dataset.
        """
        # Add dimension coordinates
        for name in self._dim_names:
            self._builder.add_coordinate(
                name,
                dimensions=(name,),
                data_type=ScalarType.INT32,
            )

        # Add non-dimension coordinates
        # TODO(Dmitriy Repin): do chunked write for non-dimensional coordinates and trace_mask
        # https://github.com/TGSAI/mdio-python/issues/587
        # The chunk size used for trace mask will be different from the _var_chunk_shape
        for i in range(len(self._coord_names)):
            self._builder.add_coordinate(
                self._coord_names[i],
                dimensions=self._coord_dim_names,
                data_type=ScalarType.FLOAT64,
                metadata=CoordinateMetadata(units_v1=self._horizontal_coord_unit),
            )

    def _add_trace_mask(self) -> None:
        """Add trace mask variables."""
        # TODO(Dmitriy Repin): do chunked write for non-dimensional coordinates and trace_mask
        # https://github.com/TGSAI/mdio-python/issues/587
        # The chunk size used for trace mask will be different from the _var_chunk_shape
        self._builder.add_variable(
            name="trace_mask",
            dimensions=self._dim_names[:-1],  # All dimensions except vertical (the last one)
            data_type=ScalarType.BOOL,
            compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),  # also default in zarr3
            coordinates=self._coord_names,
        )

    def _add_trace_headers(self, header_dtype: StructuredType) -> None:
        """Add trace mask variables."""
        # headers = StructuredType.model_validate(header_fields)

        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=self._var_chunk_shape[:-1]))
        self._builder.add_variable(
            name="headers",
            dimensions=self._dim_names[:-1],  # All dimensions except vertical (the last one)
            data_type=header_dtype,
            compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),  # also default in zarr3
            coordinates=self._coord_names,
            metadata=VariableMetadata(chunk_grid=chunk_grid),
        )

    def _add_variables(self) -> None:
        """Add custom variables.

        A virtual method that can be overwritten by subclasses to add custom variables.
        Uses the class field 'builder' to add variables to the dataset.
        """
        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=self._var_chunk_shape))
        self._builder.add_variable(
            name=self.default_variable_name,
            dimensions=self._dim_names,
            data_type=ScalarType.FLOAT32,
            compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),  # also default in zarr3
            coordinates=self._coord_names,
            metadata=VariableMetadata(chunk_grid=chunk_grid),
        )
