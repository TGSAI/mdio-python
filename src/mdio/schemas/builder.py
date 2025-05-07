"""Schema builders."""

from __future__ import annotations

from typing import Any

from mdio.schemas import NamedDimension
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset import DatasetMetadata
from mdio.schemas.v1.variable import Variable
from mdio.schemas.v1.variable import VariableMetadata


class VariableBuilder:
    """Dataset builder."""

    def __init__(self) -> None:
        self.name = None
        self.long_name = None
        self.dtype = None
        self.chunks = None
        self.dims = None
        self.coords = None
        self.compressor = None
        self.meta_dict = None

    def set_name(self, name: str) -> VariableBuilder:
        """Set variable name."""
        self.name = name
        return self

    def set_long_name(self, long_name: str) -> VariableBuilder:
        """Add long, descriptive name to the variable."""
        self.long_name = long_name
        return self

    def set_compressor(self, compressor: dict[str, Any]) -> VariableBuilder:
        """Add long, descriptive name to the variable."""
        self.compressor = compressor
        return self

    def add_dimension(self, *dimensions: str | dict[str, int]) -> VariableBuilder:
        """Add a dimension to the dataset."""
        if self.dims is None:
            self.dims = []

        if isinstance(dimensions[0], str):
            dims = list(dimensions)
        elif isinstance(dimensions[0], dict):
            dims = [
                NamedDimension(name=name, size=size)
                for dim in dimensions
                for name, size in dim.items()
            ]
        else:
            raise NotImplementedError

        self.dims.extend(dims)
        return self

    def add_coordinate(self, *names: str) -> VariableBuilder:
        """Add a coordinate to the variable."""
        if self.coords is None:
            self.coords = []

        self.coords.extend(names)
        return self

    def set_format(self, format_: str | dict[str, str]) -> VariableBuilder:
        """Set variable format."""
        if isinstance(format_, dict):
            fields = [{"name": n, "format": f} for n, f in format_.items()]
            format_ = {"fields": fields}

        self.dtype = format_
        return self

    def set_chunks(self, chunks: list[int]) -> VariableBuilder:
        """Set variable chunks."""
        if self.meta_dict is None:
            self.meta_dict = {}

        self.meta_dict["chunkGrid"] = {"configuration": {"chunkShape": chunks}}
        return self

    def set_units(self, units: dict[str, str]) -> VariableBuilder:
        """Set variable units."""
        if self.meta_dict is None:
            self.meta_dict = {}

        self.meta_dict["unitsV1"] = units
        return self

    def add_attribute(self, key: str, value: Any) -> VariableBuilder:  # noqa: ANN401
        """Add a user attribute to the variable metadata."""
        if self.meta_dict is None:
            self.meta_dict = {}

        self.meta_dict["attributes"] = {key: value}
        return self

    def build(self) -> Variable:
        """Build the dataset model."""
        if self.chunks is not None and len(self.chunks) != len(self.dims):
            msg = "Variable chunks must have same number of dimensions."
            raise ValueError(msg)

        var_kwargs = {}

        if self.meta_dict is not None:
            var_kwargs["metadata"] = VariableMetadata.model_validate(self.meta_dict)

        return Variable(
            name=self.name,
            long_name=self.long_name,
            data_type=self.dtype,
            dimensions=self.dims,
            coordinates=self.coords,
            compressor=self.compressor,
            **var_kwargs,
        )


class DatasetBuilder:
    """Dataset builder."""

    def __init__(self) -> None:
        self.variables = []
        self.name = None
        self.metadata = None

    def set_name(self, name: str) -> DatasetBuilder:
        """Set dataset name."""
        self.name = name
        return self

    def add_variable(self, variable: Variable) -> DatasetBuilder:
        """Add a variable to the dataset."""
        self.variables.append(variable)
        return self

    def add_variables(self, variables: list[Variable]) -> DatasetBuilder:
        """Add multiple variables to the dataset."""
        [self.add_variable(variable) for variable in variables]
        return self

    def set_metadata(self, metadata: DatasetMetadata) -> DatasetBuilder:
        """Add a metadata to the dataset."""
        self.metadata = metadata
        return self

    def build(self) -> Dataset:
        """Build the dataset model."""
        return Dataset(variables=self.variables, metadata=self.metadata)
