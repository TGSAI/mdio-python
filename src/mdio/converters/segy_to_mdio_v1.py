"""Conversion from SEG-Y to MDIO v1 format."""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from segy import SegyFile
from segy.arrays import HeaderArray as SegyHeaderArray
from segy.config import SegySettings
from segy.schema import SegySpec
from segy.standards.codes import MeasurementSystem as segy_MeasurementSystem
from segy.standards.fields.trace import Rev0 as TraceHeaderFieldsRev0
from xarray import Dataset as xr_Dataset
from zarr import Array as zarr_Array

from mdio.constants import fill_value_map
from mdio.converters.segy import grid_density_qc
from mdio.converters.type_converter import to_scalar_type
from mdio.converters.type_converter import to_structured_type
from mdio.core.dimension import Dimension
from mdio.core.grid import Grid
from mdio.schemas.v1.dataset_serializer import to_xarray_dataset
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.segy import blocked_io
from mdio.segy.utilities import get_grid_plan

if TYPE_CHECKING:
    from mdio.schemas.v1.dataset import Dataset


@dataclass
class StorageLocation:
    """A class to represent a local or cloud storage location for SEG-Y or MDIO files."""

    uri: str
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def storage_type(self) -> str:
        """Determine the storage type based on the URI scheme."""
        if self.uri.startswith("file://"):
            return "file"
        if self.uri.startswith("s3://"):
            return "cloud:s3"
        if self.uri.startswith("gs://"):
            return "cloud:gs"
        # Default to file storage type if no specific type is detected
        return "file"

    def exists(self) -> bool:
        """Check if the storage location exists."""
        if self.storage_type == "file":
            return Path(self.uri).exists()
        if self.storage_type.startswith("cloud:"):
            err = "Existence check for cloud storage is not implemented yet."
            raise NotImplementedError(err)
        err = f"Unsupported storage type: {self.storage_type}"
        raise ValueError(err)


def _scan_for_headers(segy_file: SegyFile) -> tuple[list[Dimension], SegyHeaderArray]:
    """Extract trace dimensions and index headers from the SEG-Y file.

    This is an expensive operation.
    It scans the SEG-Y file in chunks by using ProcessPoolExecutor
    The implementation is subject to change

    """
    # The 'grid_chunksize' is used only for grid_overrides
    # While we do not support grid override, we can set it to None
    grid_chunksize = None
    segy_dimensions, chunksize, index_headers = get_grid_plan(
        segy_file=segy_file, return_headers=True, chunksize=grid_chunksize, grid_overrides=None
    )
    return segy_dimensions, index_headers


def _get_data_coordinates(
    segy_headers: list[Dimension], mdio_template: AbstractDatasetTemplate
) -> tuple[list[Dimension], list[Dimension]]:
    """Get the data dim and non-dim coordinates from the SEG-Y headers and MDIO template.

    Select a subset of the segy_dimensions that corresponds to the MDIO dimensions
    The dimensions are ordered as in the MDIO template.
    The last dimension is always the vertical domain dimension

    Args:
        segy_headers: List of of all SEG-Y dimensions.
        mdio_template: The MDIO template to use for the conversion.

    Raises:
        ValueError: If a dimension or coordinate name from the MDIO template is not found in
                    the SEG-Y headers.

    Returns:
        A tuple containing:
            - A list of dimension coordinates (1-D arrays).
            - A list of non-dimension coordinates (N-D arrays).
    """
    dimensions_coords = []
    for coord_name in mdio_template.dimension_names[:-1]:
        coord = next((dim for dim in segy_headers if dim.name == coord_name), None)
        if coord is None:
            err = f"Dimension '{coord_name}' was not found in SEG-Y dimensions."
            raise ValueError(err)
        dimensions_coords.append(coord)
    # The last dimension returned by get_grid_plan is always the vertical dimension,
    # which is not named as in the MDIO template. Correct this.
    segy_vertical_dim = segy_headers[-1]
    segy_vertical_dim.name = mdio_template.trace_domain
    dimensions_coords.append(segy_vertical_dim)

    non_dim_coords: list[Dimension] = []
    for dim_name in mdio_template.coordinate_names:
        coord = next((c for c in segy_headers if c.name == dim_name), None)
        if coord is None:
            err = f"Coordinate '{dim_name}' not found in SEG-Y dimensions."
            raise ValueError(err)
        non_dim_coords.append(coord)

    return dimensions_coords, non_dim_coords


def populate_coordinate(
    dataset: xr_Dataset, coordinates: list[Dimension], vars_to_drop_later: list[str]
) -> xr_Dataset:
    """Populate the xarray dataset with coordinate variable."""
    for c in coordinates:
        #  If we do not have data for the coordinate variable, drop it
        if len(c.coords) == 1:
            scalar_type = to_scalar_type(c.coords.dtype)
            full_value = fill_value_map[scalar_type]
            if c.coords[0] == full_value:
                dataset = dataset.drop_vars(c.name)
                continue
        # Otherwise, populate the coordinate variable
        values = c.coords
        shape = dataset.coords[c.name].shape
        dims = dataset.coords[c.name].dims
        if shape != values.shape:
            values = values.reshape(shape)
        dataset.coords[c.name] = (dims, values)
        vars_to_drop_later.append(c.name)
    return dataset


def _get_horizontal_coordinate_unite(segy_headers: list[Dimension]) -> LengthUnitEnum | None:
    """Get the coordinate unit from the SEG-Y headers."""
    name = TraceHeaderFieldsRev0.COORDINATE_UNIT.name.upper()
    unit_hdr = next((c for c in segy_headers if c.name.upper() == name), None)
    if unit_hdr is None or len(unit_hdr.coords) == 0:
        # If the coordinate unit header is not found or empty, return None
        # This is a common case for SEG-Y files, where the coordinate unit is not specified
        return None

    if segy_MeasurementSystem(unit_hdr.coords[0]) == segy_MeasurementSystem.METERS:
        # If the coordinate unit is in meters, return "m"
        return AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))
    if segy_MeasurementSystem(unit_hdr.coords[0]) == segy_MeasurementSystem.FEET:
        # If the coordinate unit is in feet, return "ft"
        return AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT))
    err = f"Unsupported coordinate unit value: {unit_hdr.value[0]} in SEG-Y file."
    raise ValueError(err)


def _populate_coordinates_write_to_zarr(
    dataset: xr_Dataset,
    dimension_coords: list[Dimension],
    non_dim_coords: list[Dimension],
    output: StorageLocation,
) -> xr_Dataset:
    """Populate dim and non-dim coordinates in the xarray dataset and write to Zarr.

    This will write the xr Dataset with coords and dimensions, but empty traces and headers.
    """
    vars_to_drop_later = []
    # Populate the dimension coordinate variables (1-D arrays)
    dataset = populate_coordinate(
        dataset, coordinates=dimension_coords, vars_to_drop_later=vars_to_drop_later
    )

    # Populate the non-dimension coordinate variables (N-dim arrays)
    dataset = populate_coordinate(
        dataset, coordinates=non_dim_coords, vars_to_drop_later=vars_to_drop_later
    )

    dataset.to_zarr(
        store=output.uri, mode="w", write_empty_chunks=False, zarr_format=2, compute=True
    )

    # Now we can drop them to simplify chunked write of the data variable
    return dataset.drop_vars(vars_to_drop_later)


def _populate_trace_mask_write_to_zarr(
    segy_file: SegyFile,
    dimensions: list[Dimension],
    segy_index_headers: SegyHeaderArray,
    xr_sd: xr_Dataset,
    output: StorageLocation,
) -> tuple[zarr_Array, xr_Dataset]:
    """Populate and write the trace mask to Zarr, return the grid map as Zarr array."""
    # Create a grid and build live trace index
    grid = Grid(dims=dimensions)
    grid_density_qc(grid, segy_file.num_traces)
    grid.build_map(segy_index_headers)

    trace_mask = grid.live_mask

    # Populate the "trace_mask" variable in the xarray dataset and write it to Zarr
    xr_sd.trace_mask.data[:] = trace_mask
    ds_to_write = xr_sd[["trace_mask"]]
    ds_to_write.to_zarr(
        store=output.uri, mode="r+", write_empty_chunks=False, zarr_format=2, compute=True
    )
    return grid.map


def segy_to_mdio_v1(
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_location: StorageLocation,
    output_location: StorageLocation,
    overwrite: bool = False,
) -> None:
    """A function that converts a SEG-Y file to an MDIO v1 file.

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        mdio_template: The MDIO template to use for the conversion.
        input_location: The storage location of the input SEG-Y file.
        output_location: The storage location for the output MDIO v1 file.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
    """
    if overwrite and output_location.exists():
        err = f"Output location '{output_location.uri}' already exists."
        err += " Set 'overwrite' to True to overwrite it."
        raise FileExistsError(err)

    # Open a SEG-Y file according to the SegySpec
    # This could be a spec from the registryy or a custom spec
    segy_settings = SegySettings(storage_options=input_location.options)
    segy_file = SegyFile(url=input_location.uri, spec=segy_spec, settings=segy_settings)

    # Scan the SEG-Y file for headers
    # This is an memory-expensive and time-consuming read-write operation
    segy_headers, segy_index_headers = _scan_for_headers(segy_file)
    # Extract dim and non-dim coordinates according to the MDIO template
    dimension_coords, non_dim_coords = _get_data_coordinates(segy_headers, mdio_template)
    shape = [len(dim.coords) for dim in dimension_coords]
    headers = to_structured_type(segy_index_headers.dtype)

    horizontal_unit = _get_horizontal_coordinate_unite(segy_headers)
    mdio_ds: Dataset = mdio_template.build_dataset(
        name=mdio_template.name, sizes=shape, horizontal_coord_unit=horizontal_unit, headers=headers
    )
    # TODO(Dmitriy Repin): work around of the bug
    # https://github.com/TGSAI/mdio-python/issues/582
    # Do not set _FillValue for the "header" variable, which has structured data type
    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds, no_fill_var_names={"headers"})

    # Write the xr Dataset with dim and non-dim coords, but empty traces and headers.
    # We will write traces and headers in chunks later
    xr_dataset = _populate_coordinates_write_to_zarr(
        dataset=xr_dataset,
        dimension_coords=dimension_coords,
        non_dim_coords=non_dim_coords,
        output=output_location,
    )
    # Populate the live traces mask and write it to Zarr
    # Also create a grid map for the live traces
    grid_map = _populate_trace_mask_write_to_zarr(
        segy_file=segy_file,
        dimensions=dimension_coords,
        segy_index_headers=segy_index_headers,
        xr_sd=xr_dataset,
        output=output_location,
    )

    data_variable_name = mdio_template.trace_variable_name
    # This is an memory-expensive and time-consuming read-write operation
    # performed in chunks to save the memory
    blocked_io.to_zarr_v1(
        segy_file=segy_file,
        out_path=output_location.uri,
        grid_map=grid_map,
        dataset=xr_dataset,
        data_variable_name=data_variable_name,
    )
