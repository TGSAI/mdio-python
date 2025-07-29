"""Conversion from SEG-Y to MDIO v1 format."""

from dataclasses import dataclass, field
from typing import Any
from typing import Dict
import numpy as np
import xarray as xr
import zarr

from xarray import Dataset as xr_Dataset

from segy import SegyFile
from segy.config import SegySettings
from segy.schema import SegySpec
from segy.arrays import HeaderArray as SegyHeaderArray
from mdio.segy import blocked_io
from mdio.segy.utilities import get_grid_plan

from mdio.constants import UINT32_MAX
from mdio.converters.segy import grid_density_qc
from mdio.converters.type_converter import to_structured_type
from mdio.core.dimension import Dimension
from mdio.core.grid import Grid
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset_builder import _to_dictionary
from mdio.schemas.v1.dataset_serializer import to_xarray_dataset
from mdio.schemas.v1.dataset_serializer import to_zarr
from mdio.schemas.v1.stats import SummaryStatistics
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate


@dataclass
class StorageLocation:
    uri: str
    options: Dict[str, Any] = field(default_factory=dict)

def _validate_segy(coordinates: list[Dimension], templated_dataset: xr_Dataset) -> None:
    """Validate SEG-Y headers against xarray dataset created from a template."""

    coord_names = [c.name for c in coordinates]
    for name in list(templated_dataset.coords):
        if name not in coord_names:
            raise ValueError(f"Coordinate '{name}' not found in SEG-Y specification.")

def _scan_for_dims_coords_headers(
    segy_file: SegyFile,
    mdio_template: AbstractDatasetTemplate,
) -> tuple[list[Dimension], list[Dimension], SegyHeaderArray]:
    """Extract dimensions and index headers from the SEG-Y file.
    
    This is an expensive operation. 
    It scans the SEG-Y file in chunks by using ProcessPoolExecutor
    The implementation is subject to change

    """
    # TODO: Q - Where do we get an initial value? Do we need the returned value?
    # TODO: We might need to replace get_grid_plan with more efficient function
    grid_chunksize = None
    segy_dimensions, chunksize, index_headers = get_grid_plan(
        segy_file=segy_file,
        return_headers=True,
        chunksize=grid_chunksize,
        grid_overrides=None
    )

    # Select a subset of the segy_dimensions that corresponds to the MDIO dimensions
    # The dimensions are ordered as in the MDIO template.
    # The last dimension is always the vertical dimension and is treated specially
    dimensions = []
    for coord_name in mdio_template.dimension_names[:-1]:
        coord = next((dim for dim in segy_dimensions if dim.name == coord_name), None)
        if coord is None:
            err = f"Dimension '{coord_name}' not found in SEG-Y dimensions."
            raise ValueError(err)
        dimensions.append(coord)
    # The last dimension returned by get_grid_plan is always the vertical dimension, 
    # which is not named as in the MDIO template. Correct this.
    segy_vertical_dim = segy_dimensions[-1]
    segy_vertical_dim.name = mdio_template.trace_domain
    dimensions.append(segy_vertical_dim)

    coordinates = []
    for coord_name in mdio_template.coordinate_names:
        coord = next((c for c in segy_dimensions if c.name == coord_name), None)
        if coord is None:
            err = f"Coordinate '{coord_name}' not found in SEG-Y dimensions."
            raise ValueError(err)
        coordinates.append(coord)

    return dimensions, coordinates, index_headers

def _populate_dims_coords_and_write_to_zarr(
    dataset: xr_Dataset,
    dimensions: list[Dimension],
    coordinates: list[Dimension],
    output: StorageLocation) -> xr_Dataset:
    """Populate dimensions and coordinates in the xarray dataset and write to Zarr.

    This will write the xr Dataset with coords and dimensions, but empty traces and headers.
    """

    # Should we do xarray.DataArray.reset_coords first?

    vars_to_drop = list()

    for c in dimensions:
        dataset.coords[c.name] = c.coords
        vars_to_drop.append(c.name)

    for c in coordinates:
        dataset.coords[c.name] = c.coords
        vars_to_drop.append(c.name)

    dataset.to_zarr(store=output.uri,
                    mode="w",
                    write_empty_chunks=False,
                    zarr_format=2,
                    compute=True)
    
    # Now we can drop them to simplify chunked write of the data variable
    return dataset.drop_vars(vars_to_drop)

def _create_grid_map_trace_mask(grid: Grid) -> tuple[zarr.Array, zarr.Array]:
    """Create a trace mask and grid map arrays for the grid."""
    # Tuple of arrays of integer indexes along each dimension for every live traces
    live_indices = grid.header_index_arrays

    # We set dead traces to uint32 max. Should be far away from actual trace counts.
    grid_map = zarr.full(grid.shape[:-1], dtype="uint32", fill_value=UINT32_MAX)
    grid_map.vindex[live_indices] = np.arange(live_indices[0].size)

    trace_mask = zarr.zeros(grid.shape[:-1], dtype="bool")
    trace_mask.vindex[live_indices] = 1

    return grid_map, trace_mask

def _populate_trace_mask_and_write_to_zarr(segy_file: SegyFile, 
                                              dimensions: list[Dimension],
                                              index_headers: SegyHeaderArray, 
                                              xr_sd: xr_Dataset, 
                                              output: StorageLocation) -> tuple[zarr.Array, xr_Dataset]:
    """Populate and write the trace mask to Zarr, return the grid map as Zarr array.

    Returns:
      the grid map Zarr array.
    """

    # Create a grid and build live trace index
    grid = Grid(dims=dimensions)
    grid_density_qc(grid, segy_file.num_traces)
    grid.build_map(index_headers)

    # Create grid map and trace mask arrays
    grid_map, trace_mask = _create_grid_map_trace_mask(grid=grid)

    # Populate the "trace_mask" variable in the xarray dataset and write it to Zarr
    xr_sd.trace_mask.data[:] = trace_mask
    ds_to_write = xr_sd[["trace_mask"]]
    ds_to_write.to_zarr(store=output.uri, 
                        mode="r+", 
                        write_empty_chunks=False,
                        zarr_format=2,
                        compute=True)
    return grid_map

def segy_to_mdio_v1(
    input: StorageLocation,
    output: StorageLocation,
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    overwrite: bool = False,
):
    """A function that converts a SEG-Y file to an MDIO v1 file.
    """
    # Open a SEG-Y file according to the SegySpec
    # This could be a spec from the registryy or a custom spec
    segy_settings = SegySettings(storage_options=input.options)
    segy_file = SegyFile(url=input.uri, spec=segy_spec, settings=segy_settings)

    # Extract dimensions, coordinates, and index_headers
    # This is an expensive read operation performed in chunks
    dimensions, coordinates, index_headers = _scan_for_dims_coords_headers(
        segy_file, mdio_template)
    shape = [len(dim.coords) for dim in dimensions]

    # Specify the header structure for the MDIO dataset
    # TODO: Setting headers (and thus creating a variable with StructuredType data type)
    # causes the error in _populate_dims_coords_and_write_to_zarr originating
    # dask/array/core.py:1346 
    #   TypeError: Cannot cast array data from dtype([('trace_seq_num_line', '<i4'), ... 
    #   to dtype('bool') according to the rule 'unsafe'
    # I believe Dask does not support StructuredType data type.
    # Thus, we are not setting headers for now:
    headers = None
    # headers = to_structured_type(index_headers.dtype)
    # TODO: Set Units to None for now, will fix this later
    mdio_ds: Dataset = mdio_template.build_dataset(name="NONE", 
                                                   sizes=shape, 
                                                   coord_units=None,
                                                   headers=headers)
    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # Validate SEG-Y dimensions and headers against xarray dataset created from the template.
    _validate_segy(coordinates=coordinates, templated_dataset=xr_dataset)

    # Write the xr Dataset with coords and dimensions, but empty traces and headers.
    # Drop them after the write, so we can write the data variable in chunks later
    # Writing traces and headers is an memory-expensive and time-consuming operation
    # We will write them in chunks later
    xr_dataset = _populate_dims_coords_and_write_to_zarr(dataset=xr_dataset,
                                                         dimensions=dimensions,
                                                         coordinates=coordinates,
                                                         output=output)

    grid_map = _populate_trace_mask_and_write_to_zarr(segy_file=segy_file,
                                                      dimensions=dimensions,
                                                      index_headers=index_headers,
                                                      xr_sd=xr_dataset,
                                                      output=output)

    # NOTE: Maybe we should have saved the data_variable_name as a Dataset attribute?
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
