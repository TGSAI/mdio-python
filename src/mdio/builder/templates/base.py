"""Template method pattern implementation for MDIO v1 dataset template."""

from __future__ import annotations

import copy
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from mdio.builder.dataset_builder import MDIODatasetBuilder
from mdio.builder.formatting_html import template_repr_html
from mdio.builder.schemas import compressors
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.v1.units import AllUnitModel
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.schemas.v1.variable import VariableMetadata

if TYPE_CHECKING:
    from mdio.builder.schemas.v1.dataset import Dataset
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

        self._dim_names: tuple[str, ...] = ()
        self._physical_coord_names: tuple[str, ...] = ()
        self._logical_coord_names: tuple[str, ...] = ()
        self._var_chunk_shape: tuple[int, ...] = ()

        self._builder: MDIODatasetBuilder | None = None
        self._dim_sizes: tuple[int, ...] = ()
        self._units: dict[str, AllUnitModel] = {}

    def __repr__(self) -> str:
        """Return a string representation of the template."""
        return (
            f"AbstractDatasetTemplate("
            f"name={self.name!r}, "
            f"data_domain={self._data_domain!r}, "
            f"spatial_dim_names={self.spatial_dimension_names}, "
            f"dim_names={self._dim_names}, "
            f"physical_coord_names={self._physical_coord_names}, "
            f"logical_coord_names={self._logical_coord_names}, "
            f"var_chunk_shape={self._var_chunk_shape}, "
            f"dim_sizes={self._dim_sizes}, "
            f"units={self._units})"
        )

    def _repr_html_(self) -> str:
        """Return an HTML representation of the template for Jupyter notebooks."""
        return template_repr_html(self)

    def build_dataset(
        self,
        name: str,
        sizes: tuple[int, ...],
        header_dtype: StructuredType = None,
    ) -> Dataset:
        """Template method that builds the dataset.

        Args:
            name: The name of the dataset.
            sizes: The sizes of the dimensions.
            header_dtype: Optional structured headers for the dataset.

        Returns:
            Dataset: The constructed dataset
        """
        self._dim_sizes = sizes

        attributes = self._load_dataset_attributes() or {}
        attributes["defaultVariableName"] = self._default_variable_name
        self._builder = MDIODatasetBuilder(name=name, attributes=attributes)
        self._add_dimensions()
        self._add_coordinates()
        self._add_variables()
        self._add_trace_mask()

        if header_dtype is not None:
            self._add_trace_headers(header_dtype)

        return self._builder.build()

    def add_units(self, units: dict[str, AllUnitModel]) -> None:
        """Add an arbitrary number of units to the template, extending the existing ones."""
        for unit in units.values():
            if not isinstance(unit, AllUnitModel):
                msg = f"Unit {unit} is not an instance of `AllUnitModel`"
                raise ValueError(msg)
        self._units |= units

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
    def spatial_dimension_names(self) -> tuple[str, ...]:
        """Returns the names of the dimensions excluding the last axis."""
        return copy.deepcopy(self._dim_names[:-1])

    @property
    def dimension_names(self) -> tuple[str, ...]:
        """Returns the names of the dimensions."""
        return copy.deepcopy(self._dim_names)

    @property
    def physical_coordinate_names(self) -> tuple[str, ...]:
        """Returns the names of the physical (world) coordinates."""
        return copy.deepcopy(self._physical_coord_names)

    @property
    def logical_coordinate_names(self) -> tuple[str, ...]:
        """Returns the names of the logical (grid) coordinates."""
        return copy.deepcopy(self._logical_coord_names)

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        """Returns names of all coordinates."""
        return copy.deepcopy(self._physical_coord_names + self._logical_coord_names)

    @property
    def full_chunk_shape(self) -> tuple[int, ...]:
        """Returns the chunk shape for the variables."""
        return copy.deepcopy(self._var_chunk_shape)

    @full_chunk_shape.setter
    def full_chunk_shape(self, shape: tuple[int, ...]) -> None:
        """Sets the chunk shape for the variables."""
        if len(shape) != len(self._dim_sizes):
            msg = f"Chunk shape {shape} does not match dimension sizes {self._dim_sizes}"
            raise ValueError(msg)
        self._var_chunk_shape = shape

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

        A virtual method that subclasses can overwrite to return a
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

    def get_unit_by_key(self, key: str) -> AllUnitModel | None:
        """Get units by variable/dimension/coordinate name. Returns None if not found."""
        return self._units.get(key, None)

    def _add_dimensions(self) -> None:
        """Add custom dimensions.

        A virtual method that subclasses can overwrite to add custom dimensions.
        Uses the class field 'builder' to add dimensions to the dataset.
        """
        for name, size in zip(self._dim_names, self._dim_sizes, strict=True):
            self._builder.add_dimension(name, size)

    def _add_coordinates(self) -> None:
        """Add custom coordinates.

        A virtual method that subclasses can overwrite to add custom coordinates.
        Uses the class field 'builder' to add coordinates to the dataset.
        """
        # Add dimension coordinates
        for name in self._dim_names:
            self._builder.add_coordinate(
                name,
                dimensions=(name,),
                data_type=ScalarType.INT32,
                metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(name)),
            )

        # Add non-dimension coordinates
        for name in self.coordinate_names:
            self._builder.add_coordinate(
                name=name,
                dimensions=self.spatial_dimension_names,
                data_type=ScalarType.FLOAT64,
                compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),
                metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(name)),
            )

    def _add_trace_mask(self) -> None:
        """Add trace mask variables."""
        self._builder.add_variable(
            name="trace_mask",
            dimensions=self.spatial_dimension_names,
            data_type=ScalarType.BOOL,
            compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),  # also default in zarr3
            coordinates=self.coordinate_names,
        )

    def _add_trace_headers(self, header_dtype: StructuredType) -> None:
        """Add trace mask variables."""
        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=self.full_chunk_shape[:-1]))
        self._builder.add_variable(
            name="headers",
            dimensions=self.spatial_dimension_names,
            data_type=header_dtype,
            compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),  # also default in zarr3
            coordinates=self.coordinate_names,
            metadata=VariableMetadata(chunk_grid=chunk_grid),
        )

    def _add_variables(self) -> None:
        """Add custom variables.

        A virtual method that can be overwritten by subclasses to add custom variables.
        Uses the class field 'builder' to add variables to the dataset.
        """
        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=self.full_chunk_shape))
        unit = self.get_unit_by_key(self._default_variable_name)
        self._builder.add_variable(
            name=self.default_variable_name,
            dimensions=self._dim_names,
            data_type=ScalarType.FLOAT32,
            compressor=compressors.Blosc(cname=compressors.BloscCname.zstd),  # also default in zarr3
            coordinates=self.coordinate_names,
            metadata=VariableMetadata(chunk_grid=chunk_grid, units_v1=unit),
        )
