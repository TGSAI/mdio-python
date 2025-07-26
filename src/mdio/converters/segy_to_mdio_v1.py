"""Conversion from SEG-Y to MDIO v1 format."""

from dataclasses import dataclass, field
from typing import Any
from typing import Dict
import numpy as np
import xarray as xr
from segy import SegyFile
from segy.config import SegySettings
from segy.schema import SegySpec
from segy.arrays import HeaderArray as segy_HeaderArray

from xarray import Dataset as xr_Dataset
from xarray import DataArray as xr_DataArray
import zarr

from mdio.constants import UINT32_MAX
from mdio.converters.segy import grid_density_qc
from mdio.converters.type_converter import to_structured_type
from mdio.core.dimension import Dimension
from mdio.core.grid import Grid
from mdio.schemas.dtype import ScalarType, StructuredField, StructuredType
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset_builder import _to_dictionary
from mdio.schemas.v1.dataset_serializer import to_xarray_dataset
from mdio.schemas.v1.dataset_serializer import to_zarr
from mdio.schemas.v1.stats import SummaryStatistics
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.segy import blocked_io_v0
from mdio.segy import blocked_io_v1
from mdio.segy.utilities import get_grid_plan

# class StorageLocation:
#     def __init__(self, uri: str, options: dict[str, Any] = {}):
#         self.uri = uri
#         self.options = options

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

def _scan_for_dimensions_and_index_headers(
    segy_file: SegyFile,
    mdio_template: AbstractDatasetTemplate,
    
    # header_fields = segy_file.spec.trace.header.fields
    # header_dtype = np.dtype([(field.name, field.format) for field in header_fields])
    # headers = np.empty(dtype=header_dtype, shape=segy_file.num_traces)
    # Get header fields dict -> {name: dtype}
    # header_fields = {field.name: field.format for field in header_fields}: list[str]
) -> tuple[list[Dimension], segy_HeaderArray, str]:
    """Extract dimensions and index headers from the SEG-Y file.
    
    This is an expensive operation and scans the SEG-Y file in chunks
    by using ProcessPoolExecutor
    """

    # Q: Where do we get an initial value. 
    # We might need to rewrite get_grid_plan
    # BUG: segy_dimensions for coordinates do not have values 
    grid_chunksize = None
    segy_dimensions, chunksize, index_headers = get_grid_plan(
        segy_file=segy_file,
        return_headers=True,
        chunksize=grid_chunksize,
        grid_overrides=None
    )

    # Select a subset of the dimensions.
    # The last dimension is always the vertical dimension and is treated specially
    dimensions = []
    #TODO: expose mdio_template._dim_names as a public method
    mdio_dimension_names = mdio_template._dim_names
    for coord_name in mdio_dimension_names[:-1]:
        coord = next((dim for dim in segy_dimensions if dim.name == coord_name), None)
        if coord is None:
            err = f"Dimension '{coord_name}' not found in SEG-Y dimensions."
            raise ValueError(err)
        dimensions.append(coord)
    # The last dimension returned by get_grid_plan is always the vertical dimension, 
    # which is not named as in the MDIO template. Correct this.
    segy_vertical_dim = segy_dimensions[-1]
    #TODO: expose mdio_template._trace_domain as a public method
    domain = mdio_template._trace_domain
    segy_vertical_dim.name = domain
    dimensions.append(segy_vertical_dim)

    coordinates = []
    #TODO: expose mdio_template._coord_names as a public method
    #TODO: expose mdio_template._coord_dim_names as a public method
    # mdio_coord_names = mdio_template._coord_dim_names + mdio_template._coord_names
    mdio_coord_names = mdio_template._coord_names
    for coord_name in mdio_coord_names:
        coord = next((c for c in segy_dimensions if c.name == coord_name), None)
        if coord is None:
            err = f"Coordinate '{coord_name}' not found in SEG-Y dimensions."
            raise ValueError(err)
        coordinates.append(coord)

    return dimensions, coordinates, index_headers

def _get_dimension_sizes(segy_dimensions: list[Dimension], veritcal_dimension_name: str, mdio_dimension_names: list[str]) -> list[int]:
    """Get the sizes of the dimensions.

    Args:
        segy_dimensions: The list of SEG-Y dimensions.
        sample_dimension_name: The name of the sample dimension.
        mdio_dimension_names: The list of MDIO dimension names.

    Returns:
        list[int]: The sizes of the dimensions.
    """
    # SPECIAL CASE: The vertical dimension ('time' or 'depth') is not found in the segy_dimensions
    # src\segy\standards\fields\trace.py
    # The non-standard name of that dimension is in veritcal_dimension_name.
    dimension_names = list(mdio_dimension_names[:-1]) + [veritcal_dimension_name]
    sizes = []
    for dim_name in dimension_names:
        # sz = len(dim.coords) for dim in segy_dimensions if dim.name == dim_name
        dim = next((dim for dim in segy_dimensions if dim.name == dim_name), None)
        if dim is None:
            err = f"Dimension '{dim_name}' not found in SEG-Y dimensions."
            raise ValueError(err)
        sizes.append(len(dim.coords))
    return sizes

def _add_coordinates_to_dataset(
    dataset: xr_Dataset,
    dimensions: list[Dimension],
    coordinates: list[Dimension]
) -> None:
    """Add coordinates to the xarray dataset."""
    for c in dimensions:
        dataset.coords[c.name] = c.coords

    for c in coordinates:
        dataset.coords[c.name] = c.coords


def _get_trace_mask(grid: Grid):

    live_indices = grid.header_index_arrays

    # We set dead traces to uint32 max. Should be far away from actual trace counts.
    grid_map = zarr.full(grid.shape[:-1], dtype="uint32", fill_value=UINT32_MAX)
    grid_map.vindex[live_indices] = np.arange(live_indices[0].size)

    trace_mask = zarr.zeros(grid.shape[:-1], dtype="bool")
    trace_mask.vindex[live_indices] = 1

    return grid_map, trace_mask

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
    segy_settings = SegySettings(storage_options=input.options)
    segy_file = SegyFile(url=input.uri, spec=segy_spec, settings=segy_settings)

    # Extract dimensions, index_headers (containing coords) and (optionally) units from the SEG-Y file
    dimensions, coordinates, index_headers = _scan_for_dimensions_and_index_headers(
        segy_file, mdio_template)
    dimension_sizes = [len(dim.coords) for dim in dimensions]

    headers = None
    # TODO: Setting headers (and thus creating a variable with StructuredType data type)
    # causes the error in dask/array/core.py:1346
    #   TypeError: Cannot cast array data from dtype([('trace_seq_num_line', '<i4'), ... 
    #   to dtype('bool') according to the rule 'unsafe'
    #
    # headers = to_structured_type(index_headers.dtype)
    # headers = StructuredType(
    #     fields=[
    #             StructuredField(name="cdp_x", format=ScalarType.INT32),
    #             StructuredField(name="cdp_y", format=ScalarType.INT32),
    #             StructuredField(name="elevation", format=ScalarType.FLOAT16),
    #             StructuredField(name="some_scalar", format=ScalarType.FLOAT16),
    #     ]
    # )
    
    # Create an empty MDIO Zarr dataset based on the specified MDIO template
    # TODO: Set Units to None for now, will fix this later
    mdio_ds: Dataset = mdio_template.build_dataset(name="NONE", 
                                                   sizes=dimension_sizes, 
                                                   coord_units=None,
                                                   headers=headers)
    data_variable_name = mdio_template.get_data_variable_name()
    xr_sd: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # Validate SEG-Y dimensions and headers against xarray dataset created from the template.
    _validate_segy(coordinates=coordinates, templated_dataset=xr_sd)

    # Create a grid and build trace map and live mask.
    grid = Grid(dims=dimensions)
    grid_density_qc(grid, segy_file.num_traces)
    grid.build_map(index_headers)

    # Add coordinates values to the xarray dataset
    _add_coordinates_to_dataset(dataset=xr_sd, dimensions=dimensions, coordinates=coordinates)
    # We will write coords and dimensions, but empty traces and headers
    to_zarr(dataset=xr_sd, store=output.uri, storage_options=output.options)

    # header_array / metadata_array: Handle for zarr.Array we are writing trace headers
    data: xr_DataArray = xr_sd[data_variable_name]

    # Create and populate trace mask
    grid_map, trace_mask = _get_trace_mask(grid=grid)
    xr_sd.trace_mask.data[:] = trace_mask

    # Get subset of the dataset containing only "trace_mask"
    #TODO: Should we write the trace mask inside of blocked_io_v1.to_zarr?
    #TODO: Should we clear the coords from ds_to_write?
    ds_to_write = xr_sd[["trace_mask"]]
    ds_to_write.to_zarr(output.uri, mode="r+", write_empty_chunks=False)

    stats: SummaryStatistics = blocked_io_v1.to_zarr(
        segy_file = segy_file,
        out_path = output.uri,
        grid_map = grid_map,
        dataset = xr_sd,
        data_variable_name = data_variable_name
    )

    # stats = blocked_io_v0.to_zarr(
    #     segy_file=segy_file,
    #     grid=grid,
    #     data_array=data,
    #     header_array=headers,
    #     mdio_path_or_buffer=output.uri,
    # )
    # stats:
    # {"mean": glob_mean, "std": glob_std, "rms": glob_rms, "min": glob_min, "max": glob_max}

    # TODO: Write actual stats to the MDIO Zarr dataset
