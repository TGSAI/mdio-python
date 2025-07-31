"""Conversion from SEG-Y to MDIO v1 format."""
from collections.abc import Sequence

from dataclasses import dataclass, field
from typing import Any
from typing import Dict

from zarr import Array as zarr_Array
from xarray import Dataset as xr_Dataset

from segy import SegyFile
from segy.config import SegySettings
from segy.schema import SegySpec
from segy.schema import HeaderField
from segy.arrays import HeaderArray as SegyHeaderArray
from mdio.converters.type_converter import to_scalar_type
from mdio.schemas.v1.templates.template_registry import TemplateRegistry
from mdio.segy import blocked_io
from mdio.segy.utilities import get_grid_plan
from segy.standards import get_segy_standard

from mdio.constants import UINT32_MAX
from mdio.converters.segy import grid_density_qc
from mdio.constants import fill_value_map
from mdio.core.dimension import Dimension
from mdio.core.grid import Grid
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset_serializer import to_xarray_dataset
from mdio.schemas.v1.stats import SummaryStatistics
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate



@dataclass
class StorageLocation:
    """A class to represent a local or cloud storage location for SEG-Y or MDIO files."""
    uri: str
    options: Dict[str, Any] = field(default_factory=dict)

def customize_segy_specs(
    segy_spec: SegySpec,
    index_bytes: Sequence[int] | None = None,
    index_names: Sequence[str] | None = None,
    index_types: Sequence[str] | None = None,
) -> SegySpec:
    """Customize SEG-Y specifications with user-defined index fields."""
    if not index_bytes:
        # No customization
        return segy_spec

    index_names = index_names or [f"dim_{i}" for i in range(len(index_bytes))]
    index_types = index_types or ["int32"] * len(index_bytes)

    if not (len(index_names) == len(index_bytes) == len(index_types)):
        raise ValueError("All index fields must have the same length.")

    # Index the dataset using a spec that interprets the user provided index headers.
    index_fields = []
    for name, byte, format_ in zip(index_names, index_bytes, index_types, strict=True):
        index_fields.append(HeaderField(name=name, byte=byte, format=format_))

    custom_spec = segy_spec.customize(trace_header_fields=index_fields)
    return custom_spec

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
        segy_file=segy_file,
        return_headers=True,
        chunksize=grid_chunksize,
        grid_overrides=None
    )
    return segy_dimensions, index_headers


def _get_data_coordinates(segy_headers: list[Dimension], 
                          mdio_template: AbstractDatasetTemplate) -> tuple[list[Dimension], list[Dimension]]:
    """Get the data dim and non-dim coordinates from the SEG-Y headers and MDIO template.
        
    Select a subset of the segy_dimensions that corresponds to the MDIO dimensions
    The dimensions are ordered as in the MDIO template. 
    The last dimension is always the vertical domain dimension
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


def populate_coordinate(dataset: xr_Dataset, 
                        coordinates: list[Dimension], 
                        vars_to_drop_later: list[str]) -> None:
    """Populate the xarray dataset with coordinate variable."""
    for c in coordinates:
        #  If we do not have data for the coordinate variable, drop it
        if len(c.coords) == 1:
            scalar_type = to_scalar_type(c.coords.dtype)
            full_value = fill_value_map[scalar_type]
            if c.coords[0] == full_value:
                dataset.drop_vars(c.name)
                continue
        # Otherwise, populate the coordinate variable
        values = c.coords
        shape = dataset.coords[c.name].shape
        dims = dataset.coords[c.name].dims
        if shape != values.shape:
            values = values.reshape(shape)
        dataset.coords[c.name] = (dims, values)
        vars_to_drop_later.append(c.name)


def _populate_coordinates_write_to_zarr(
    dataset: xr_Dataset,
    dimension_coords: list[Dimension],
    non_dim_coords: list[Dimension],
    output: StorageLocation) -> xr_Dataset:
    """Populate dim and non-dim coordinates in the xarray dataset and write to Zarr.

    This will write the xr Dataset with coords and dimensions, but empty traces and headers.
    """
    vars_to_drop_later = list()
    # Populate the dimension coordinate variables (1-D arrays)
    populate_coordinate(dataset,
                        coordinates=dimension_coords,
                        vars_to_drop_later=vars_to_drop_later)

    # Populate the non-dimension coordinate variables (N-dim arrays)
    populate_coordinate(dataset,
                        coordinates=non_dim_coords,
                        vars_to_drop_later=vars_to_drop_later)

    dataset.to_zarr(store=output.uri,
                    mode="w",
                    write_empty_chunks=False,
                    zarr_format=2,
                    compute=True)

    # Now we can drop them to simplify chunked write of the data variable
    return dataset.drop_vars(vars_to_drop_later)

def _populate_trace_mask_write_to_zarr(segy_file: SegyFile,
                                        dimensions: list[Dimension],
                                        segy_index_headers: SegyHeaderArray,
                                        xr_sd: xr_Dataset,
                                        output: StorageLocation) -> tuple[zarr_Array, xr_Dataset]:
    """Populate and write the trace mask to Zarr, return the grid map as Zarr array.

    Returns:
      the grid map Zarr array.
    """

    # Create a grid and build live trace index
    grid = Grid(dims=dimensions)
    grid_density_qc(grid, segy_file.num_traces)
    grid.build_map(segy_index_headers)

    trace_mask = grid.live_mask

    # Populate the "trace_mask" variable in the xarray dataset and write it to Zarr
    xr_sd.trace_mask.data[:] = trace_mask
    ds_to_write = xr_sd[["trace_mask"]]
    ds_to_write.to_zarr(store=output.uri,
                        mode="r+",
                        write_empty_chunks=False,
                        zarr_format=2,
                        compute=True)
    return grid.map


def segy_to_mdio_v1_customized(
    segy_spec: str,
    mdio_template: str,
    input: StorageLocation,
    output: StorageLocation,
    index_bytes: Sequence[int] | None = None,
    index_names: Sequence[str] | None = None,
    index_types: Sequence[str] | None = None,
    overwrite: bool = False,
):
    """A function that converts a SEG-Y file to an MDIO v1 file.

    This function takes in various variations of input parameters and normalizes
    them, performs necessary customizations before calling segy_2_mdio() to
    perform the conversion from SEG-Y and MDIO v1 formats.
    """
    # Retrieve the SEG-Y specifications either from a registry or a storage location
    try:
        segy_spec = float(segy_spec)
        segy_spec = get_segy_standard(segy_spec)
    except:
        err = f"SEG-Y spec '{segy_spec}' is not registered."
        raise ValueError(err)

    # Retrieve MDIO template either from a registry or a storage location
    if not TemplateRegistry().is_registered(mdio_template):
        err = f"MDIO template '{mdio_template}' is not registered."
        raise ValueError(err)
    mdio_template = TemplateRegistry().get(mdio_template)

    # Customize the SEG-Y specs, if customizations are provided
    segy_spec = customize_segy_specs(
        segy_spec, index_bytes=index_bytes, index_names=index_names, index_types=index_types
    )

    # Proceed with the conversion
    segy_to_mdio_v1(
        segy_spec=segy_spec,
        mdio_template=mdio_template,
        input=input,
        output=output,
        overwrite=overwrite,
    )


def segy_to_mdio_v1(
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input: StorageLocation,
    output: StorageLocation,
    overwrite: bool = False,
):
    """A function that converts a SEG-Y file to an MDIO v1 file.
    """
    # Open a SEG-Y file according to the SegySpec
    # This could be a spec from the registryy or a custom spec
    segy_settings = SegySettings(storage_options=input.options)
    segy_file = SegyFile(url=input.uri, spec=segy_spec, settings=segy_settings)

    # Scan the SEG-Y file for headers
    # This is an memory-expensive and time-consuming read-write operation
    segy_headers, segy_index_headers = _scan_for_headers(segy_file)
    # Extract dim and non-dim coordinates according to the MDIO template
    dimension_coords, non_dim_coords = _get_data_coordinates(segy_headers, mdio_template)
    shape = [len(dim.coords) for dim in dimension_coords]

    # Specify the header structure for the MDIO dataset
    #   TODO: Setting headers (and thus creating a variable with StructuredType data type)
    #   causes the error in _populate_dims_coords_and_write_to_zarr originating
    #   Thus, we are not setting headers for now:
    headers = None
    # headers = to_structured_type(index_headers.dtype)
    #
    # TODO: Set Units to None for now, will fix this later
    mdio_ds: Dataset = mdio_template.build_dataset(name=mdio_template.name,
                                                   sizes=shape,
                                                   coord_units=None,
                                                   headers=headers)
    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # Write the xr Dataset with dim and non-dim coords, but empty traces and headers.
    # We will write traces and headers in chunks later
    xr_dataset = _populate_coordinates_write_to_zarr(dataset=xr_dataset,
                                                     dimension_coords=dimension_coords,
                                                     non_dim_coords=non_dim_coords,
                                                     output=output)
    # Populate the live traces mask and write it to Zarr
    # Also create a grid map for the live traces
    grid_map = _populate_trace_mask_write_to_zarr(segy_file=segy_file,
                                                  dimensions=dimension_coords,
                                                  segy_index_headers=segy_index_headers,
                                                  xr_sd=xr_dataset,
                                                  output=output)

    data_variable_name = mdio_template.trace_variable_name
    # This is an memory-expensive and time-consuming read-write operation
    # performed in chunks to save the memory
    stats: SummaryStatistics = blocked_io.to_zarr(
        segy_file = segy_file,
        out_path = output.uri,
        grid_map = grid_map,
        dataset = xr_dataset,
        data_variable_name = data_variable_name
    )

