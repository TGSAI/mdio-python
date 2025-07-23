"""Conversion from SEG-Y to MDIO v1 format."""

from segy import SegyFile
from segy.config import SegySettings
from segy.schema import SegySpec
from xarray import Dataset as xr_Dataset

from mdio.converters.segy import grid_density_qc
from mdio.converters.segy_to_mdio_v1_custom import StorageLocation
from mdio.core.grid import Grid
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset_serializer import to_xarray_dataset
from mdio.schemas.v1.dataset_serializer import to_zarr
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.segy import blocked_io
from mdio.segy.utilities import get_grid_plan


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
    segy_settings = SegySettings(storage_options=input.storage_options)
    segy_file = SegyFile(url=input.uri, spec=segy_spec, settings=segy_settings)

    # Extract dataset dimensions and (optionally) units from the SEG-Y file
    grid_chunksize = (
        None  # Q: Where do we get an initial value. Do we need to expose them from mdio_template?
    )
    # Do we need the returned chunksize? mdio_template contains a predefined one.
    dimensions, chunksize, index_headers = get_grid_plan(
        segy_file=segy_file,
        return_headers=True,
        chunksize=grid_chunksize,
        grid_overrides=None,  # TODO: For now it is None, but can be set later.
    )

    # Validate the specified MDIO template matches the SegySpec (how?)
    # We need to check if the MDIO template has the required keys (dimension names, and coordinates).
    # We also need to validate if the SegySpec has these keys because MDIO will try to read from those.

    # Create a grid and build trace map and live mask.
    grid = Grid(dims=dimensions)
    grid_density_qc(grid, segy_file.num_traces)
    grid.build_map(index_headers)

    # Create an empty MDIO Zarr dataset based on the specified MDIO template
    # TODO: Set Units to None for now, will fix this later
    mdio_ds: Dataset = mdio_template.build_dataset(sizes=dimensions, coord_units=None)
    xr_sd: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)
    # Add coordinates and dimensions to the xarray dataset to write them
    to_zarr(dataset=xr_sd, store=output.uri, storage_options=output.options)

    # Write traces to the MDIO Zarr dataset
    # Currently the name of the variable in the dataset for the data volume is:
    # e.g. "StackedAmplitude" for 2D post-stack depth
    # e.g. "StackedAmplitude" for 3D post-stack depth
    # e.g. "AmplitudeCDP" for 3D pre-stack CPD depth
    # e.g. "AmplitudeShot" for 3D pre-stack time Shot gathers
    variable_name = "Amplitude"  # TODO: Use the proper variable name
    stats = blocked_io.to_zarr(
        segy_file=segy_file,
        grid=grid,
        data_array=xr_sd["Amplitude"],
        header_array=None,  # TODO: where do we get the header array from?
    )
    # stats:
    # {"mean": glob_mean, "std": glob_std, "rms": glob_rms, "min": glob_min, "max": glob_max}

    # TODO: Write actual stats to the MDIO Zarr dataset
