"""Conversion from SEG-Y to MDIO v1 format."""

from dataclasses import dataclass, field
from typing import Any
from typing import Dict

from zarr import Array as zarr_Array
from xarray import Dataset as xr_Dataset

from segy import SegyFile
from segy.config import SegySettings
from segy.schema import SegySpec
from segy.arrays import HeaderArray as SegyHeaderArray

from mdio.segy import blocked_io
from mdio.segy.utilities import get_grid_plan
from mdio.constants import UINT32_MAX
from mdio.converters.segy import grid_density_qc
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


def _populate_coordinates_write_to_zarr(
    dataset: xr_Dataset,
    dimension_coords: list[Dimension],
    non_dim_coords: list[Dimension],
    mdio_template: AbstractDatasetTemplate,
    output: StorageLocation) -> xr_Dataset:
    """Populate dim and non-dim coordinates in the xarray dataset and write to Zarr.

    This will write the xr Dataset with coords and dimensions, but empty traces and headers.
    """
    vars_to_drop = list()

    # TODO: Bug - the coordinate dimension variable is not defined
    # Populate the dimension coordinate variables (1-D arrays)
    # for c in dimensions:
    #     dataset.coords[c.name] = c.coords
    #     vars_to_drop.append(c.name)

    # Populate the non-dimension coordinate variables (N-dim arrays)
    # for c in coordinates:
    #     # Need N-dim arrays to populate the coordinates
    #     dataset.coords[c.name] = c.coords
    #     vars_to_drop.append(c.name)

    dataset.to_zarr(store=output.uri,
                    mode="w",
                    write_empty_chunks=False,
                    zarr_format=2,
                    compute=True)

    # Now we can drop them to simplify chunked write of the data variable
    return dataset.drop_vars(vars_to_drop)

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
    # Extract dime and non-dim coordinates according to the MDIO template
    dimension_coords, non_dim_coords = _get_data_coordinates(segy_headers, mdio_template)
    shape = [len(dim.coords) for dim in dimension_coords]

    # Specify the header structure for the MDIO dataset
    #   TODO: Setting headers (and thus creating a variable with StructuredType data type)
    #   causes the error in _populate_dims_coords_and_write_to_zarr originating
    #   Thus, we are not setting headers for now:
    headers = None
    # headers = to_structured_type(index_headers.dtype)
    # TODO: Set Units to None for now, will fix this later
    mdio_ds: Dataset = mdio_template.build_dataset(name="NONE",
                                                   sizes=shape,
                                                   coord_units=None,
                                                   headers=headers)
    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # Write the xr Dataset with dim and non-dim coords, but empty traces and headers.
    # We will write traces and headers in chunks later
    xr_dataset = _populate_coordinates_write_to_zarr(dataset=xr_dataset,
                                                     dimension_coords=dimension_coords,
                                                     non_dim_coords=non_dim_coords,
                                                     mdio_template=mdio_template,
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

    # TODO: Write actual stats to the MDIO Zarr dataset
