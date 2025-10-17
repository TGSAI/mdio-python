"""Builder pattern implementation for MDIO v1 schema models."""

from datetime import UTC
from datetime import datetime
from enum import Enum
from enum import auto
from typing import Any

from mdio import __version__
from mdio.builder.schemas.compressors import ZFP
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.dimension import NamedDimension
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.v1.dataset import Dataset
from mdio.builder.schemas.v1.dataset import DatasetMetadata
from mdio.builder.schemas.v1.variable import Coordinate
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.schemas.v1.variable import Variable
from mdio.builder.schemas.v1.variable import VariableMetadata


class _BuilderState(Enum):
    """States for the template builder."""

    INITIAL = auto()
    HAS_DIMENSIONS = auto()
    HAS_COORDINATES = auto()
    HAS_VARIABLES = auto()


def _get_named_dimension(dimensions: list[NamedDimension], name: str, size: int | None = None) -> NamedDimension | None:
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
        self._metadata = DatasetMetadata(
            name=name,
            api_version=__version__,
            created_on=datetime.now(UTC),
            attributes=attributes,
        )
        self._dimensions: list[NamedDimension] = []
        self._coordinates: list[Coordinate] = []
        self._variables: list[Variable] = []
        self._state = _BuilderState.INITIAL
        self._unnamed_variable_counter = 0

    def add_dimension(self, name: str, size: int) -> "MDIODatasetBuilder":
        """Add a dimension.

        This function be called at least once before adding coordinates or variables.

        Args:
            name: Name of the dimension
            size: Size of the dimension

        Raises:
            ValueError: If 'name' is not a non-empty string.
                        if the dimension is already defined.

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

    def add_coordinate(  # noqa: PLR0913
        self,
        name: str,
        *,
        long_name: str = None,
        dimensions: tuple[str, ...],
        data_type: ScalarType,
        compressor: Blosc | ZFP | None = None,
        metadata: CoordinateMetadata | None = None,
    ) -> "MDIODatasetBuilder":
        """Add a coordinate after adding at least one dimension.

        This function must be called after all required dimensions are added via add_dimension().
        This call will create a coordinate variable.

        Args:
            name: Name of the coordinate
            long_name: Optional long name for the coordinate
            dimensions: List of dimension names that the coordinate is associated with
            data_type: Data type for the coordinate (defaults to FLOAT32)
            compressor: Compressor used for the variable (defaults to None)
            metadata: Optional metadata information for the coordinate

        Raises:
            ValueError: If no dimensions have been added yet.
                        If 'name' is not a non-empty string.
                        If 'dimensions' is not a non-empty list.
                        If the coordinate is already defined.
                        If any referenced dimension is not already defined.

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

        coord = Coordinate(
            name=name,
            long_name=long_name,
            dimensions=named_dimensions,
            compressor=compressor,
            data_type=data_type,
            metadata=metadata,
        )
        self._coordinates.append(coord)

        # Add a coordinate variable to the dataset
        var_metadata = None if coord.metadata is None else VariableMetadata(**coord.metadata.model_dump())
        self.add_variable(
            name=coord.name,
            long_name=coord.long_name,
            dimensions=dimensions,  # dimension names (list[str])
            data_type=coord.data_type,
            compressor=compressor,
            coordinates=[name],  # Use the coordinate name as a reference
            metadata=var_metadata,
        )

        self._state = _BuilderState.HAS_COORDINATES
        return self

    def add_variable(  # noqa: PLR0913
        self,
        name: str,
        *,
        long_name: str = None,
        dimensions: tuple[str, ...],
        data_type: ScalarType | StructuredType,
        compressor: Blosc | ZFP | None = None,
        coordinates: tuple[str, ...] | None = None,
        metadata: VariableMetadata | None = None,
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
            metadata: Optional metadata information for the variable

        Raises:
            ValueError: If no dimensions have been added yet.
                        If 'name' is not a non-empty string.
                        If 'dimensions' is not a non-empty list.
                        If the variable is already defined.
                        If any referenced dimension is not already defined.
                        If any referenced coordinate is not already defined.

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
            if name == "trace_mask":  # coverage: exclude
                msg += (
                    "Error occurred while adding special case variable `trace_mask`."
                    "This is likely an issue with how a custom template has been defined."
                    "Please check the documentation https://github.com/TGSAI/mdio-python/issues/718 "
                    "for more information."
                    # TODO(Anybody): Update the link to point to the appropriate documentation when it is available.
                    # https://github.com/TGSAI/mdio-python/issues/718
                )
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

        coordinate_objs: list[Coordinate] = []
        # Validate that all referenced coordinates are already defined
        if coordinates is not None:
            for coord in coordinates:
                c: Coordinate = next((c for c in self._coordinates if c.name == coord), None)
                if c is not None:
                    coordinate_objs.append(c)
                else:
                    msg = f"Pre-existing coordinate named {coord!r} is not found"
                    raise ValueError(msg)

        # If this is a dimension coordinate variable, embed the Coordinate into it
        if coordinates is not None and len(coordinates) == 1 and coordinates[0] == name:
            coordinates = coordinate_objs

        var = Variable(
            name=name,
            long_name=long_name,
            dimensions=named_dimensions,
            data_type=data_type,
            compressor=compressor,
            coordinates=coordinates,
            metadata=metadata,
        )
        self._variables.append(var)

        self._state = _BuilderState.HAS_VARIABLES
        return self

    def build(self) -> Dataset:
        """Build the final dataset.

        This function must be called after at least one dimension is added via add_dimension().
        It will create a Dataset object with all added dimensions, coordinates, and variables.

        Raises:
            ValueError: If no dimensions have been added yet.

        Returns:
            Dataset: The built dataset with all added dimensions, coordinates, and variables.
        """
        if self._state == _BuilderState.INITIAL:
            msg = "Must add at least one dimension before building"
            raise ValueError(msg)

        return Dataset(variables=self._variables, metadata=self._metadata)

    def __repr__(self) -> str:
        """Return a string representation of the builder."""
        dim_names = [d.name for d in self._dimensions]
        coord_names = [c.name for c in self._coordinates]
        var_names = [v.name for v in self._variables]
        return (
            f"MDIODatasetBuilder("
            f"name={self._metadata.name!r}, "
            f"state={self._state.name}, "
            f"dimensions={dim_names}, "
            f"coordinates={coord_names}, "
            f"variables={var_names})"
        )

    def _repr_html_(self) -> str:
        """Return an HTML representation of the builder for Jupyter notebooks."""
        dim_rows = ""
        td_style = "padding: 8px; text-align: left; border-bottom: 1px solid rgba(128, 128, 128, 0.2);"
        for dim in self._dimensions:
            dim_rows += f"<tr><td style='{td_style}'>{dim.name}</td><td style='{td_style}'>{dim.size}</td></tr>"

        coord_rows = ""
        for coord in self._coordinates:
            dims_str = ", ".join(d.name for d in coord.dimensions)
            coord_rows += (
                f"<tr><td style='{td_style}'>{coord.name}</td>"
                f"<td style='{td_style}'>{dims_str}</td>"
                f"<td style='{td_style}'>{coord.data_type}</td></tr>"
            )

        var_rows = ""
        for var in self._variables:
            dims_str = ", ".join(d.name for d in var.dimensions)
            var_rows += (
                f"<tr><td style='{td_style}'>{var.name}</td>"
                f"<td style='{td_style}'>{dims_str}</td>"
                f"<td style='{td_style}'>{var.data_type}</td></tr>"
            )

        box_style = (
            "font-family: monospace; border: 1px solid rgba(128, 128, 128, 0.3); "
            "border-radius: 5px; padding: 15px; max-width: 1000px;"
        )
        header_style = (
            "padding: 10px; margin: -15px -15px 15px -15px; border-bottom: 2px solid rgba(128, 128, 128, 0.3);"
        )
        summary_style = "cursor: pointer; font-weight: bold; margin-bottom: 8px;"
        summary_style_2 = "cursor: pointer; font-weight: bold; margin: 15px 0 8px 0;"

        no_dims = (
            '<tr><td colspan="2" style="padding: 8px; opacity: 0.5; text-align: left;">No dimensions added</td></tr>'  # noqa: E501
        )
        no_coords = (
            '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No coordinates added</td></tr>'  # noqa: E501
        )
        no_vars = (
            '<tr><td colspan="3" style="padding: 8px; opacity: 0.5; text-align: left;">No variables added</td></tr>'  # noqa: E501
        )

        return f"""
        <div style="{box_style}">
            <div style="{header_style}">
                <strong style="font-size: 1.1em;">MDIODatasetBuilder</strong>
            </div>
            <div style="margin-bottom: 15px;">
                <strong>Name:</strong> {self._metadata.name}<br>
                <strong>State:</strong> {self._state.name}<br>
                <strong>API Version:</strong> {self._metadata.api_version}<br>
                <strong>Created:</strong> {self._metadata.created_on.strftime("%Y-%m-%d %H:%M:%S UTC")}
            </div>
            <details open>
                <summary style="{summary_style}">▸ Dimensions ({len(self._dimensions)})</summary>
                <div style="margin-left: 20px;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);">
                                <th style="text-align: left; padding: 8px; font-weight: 600;">Name</th>
                                <th style="text-align: left; padding: 8px; font-weight: 600;">Size</th>
                            </tr>
                        </thead>
                        <tbody>
                            {dim_rows if dim_rows else no_dims}
                        </tbody>
                    </table>
                </div>
            </details>
            <details open>
                <summary style="{summary_style_2}">▸ Coordinates ({len(self._coordinates)})</summary>
                <div style="margin-left: 20px;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);">
                                <th style="text-align: left; padding: 8px; font-weight: 600;">Name</th>
                                <th style="text-align: left; padding: 8px; font-weight: 600;">Dimensions</th>
                                <th style="text-align: left; padding: 8px; font-weight: 600;">Type</th>
                            </tr>
                        </thead>
                        <tbody>
                            {coord_rows if coord_rows else no_coords}
                        </tbody>
                    </table>
                </div>
            </details>
            <details open>
                <summary style="{summary_style_2}">▸ Variables ({len(self._variables)})</summary>
                <div style="margin-left: 20px;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="border-bottom: 2px solid rgba(128, 128, 128, 0.4);">
                                <th style="text-align: left; padding: 8px; font-weight: 600;">Name</th>
                                <th style="text-align: left; padding: 8px; font-weight: 600;">Dimensions</th>
                                <th style="text-align: left; padding: 8px; font-weight: 600;">Type</th>
                            </tr>
                        </thead>
                        <tbody>
                            {var_rows if var_rows else no_vars}
                        </tbody>
                    </table>
                </div>
            </details>
        </div>
        """
