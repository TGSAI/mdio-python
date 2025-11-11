"""Creating MDIO v1 datasets."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING

from mdio.api.io import _normalize_path
from mdio.api.io import open_mdio
from mdio.api.io import to_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.converters.segy import populate_dim_coordinates
from mdio.converters.type_converter import to_structured_type
from mdio.core.grid import Grid

if TYPE_CHECKING:
    from pathlib import Path

    from segy.schema import HeaderSpec
    from upath import UPath
    from xarray import Dataset as xr_Dataset

    from mdio.builder.schemas import Dataset
    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.core.dimension import Dimension


def create_empty(
    mdio_template: AbstractDatasetTemplate | str,
    dimensions: list[Dimension],
    output_path: UPath | Path | str | None,
    headers: HeaderSpec | None = None,
    overwrite: bool = False,
) -> xr_Dataset:
    """A function that creates an empty MDIO v1 file with known dimensions.

    Args:
        mdio_template: The MDIO template or template name to use to define the dataset structure.
        dimensions: The dimensions of the MDIO file.
        output_path: The universal path for the output MDIO v1 file.
        headers: The SEG-Y trace headers that are important to the Dataset. Defaults to None.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.

    Returns:
        The output MDIO dataset.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
    """
    output_path = _normalize_path(output_path)

    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    header_dtype = to_structured_type(headers.dtype) if headers else None
    grid = Grid(dims=dimensions)
    if isinstance(mdio_template, str):
        # A template name is passed in. Get a unit-unaware template from registry
        mdio_template = TemplateRegistry().get(mdio_template)
    # Build the dataset using the template
    mdio_ds: Dataset = mdio_template.build_dataset(name=mdio_template.name, sizes=grid.shape, header_dtype=header_dtype)

    # Convert to xarray dataset
    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # Populate coordinates using the grid
    # For empty datasets, we only populate dimension coordinates
    drop_vars_delayed = []
    xr_dataset, drop_vars_delayed = populate_dim_coordinates(xr_dataset, grid, drop_vars_delayed=drop_vars_delayed)

    if headers:
        # Since the headers were provided, the user wants to export to SEG-Y
        # Add a dummy segy_file_header variable used to export to SEG-Y
        xr_dataset["segy_file_header"] = ((), "")

    # Create the Zarr store with the correct structure but with empty arrays
    if output_path is not None:
        to_mdio(xr_dataset, output_path=output_path, mode="w", compute=False)

    # Write the dimension coordinates and trace mask
    xr_dataset = xr_dataset[drop_vars_delayed + ["trace_mask"]]

    if output_path is not None:
        to_mdio(xr_dataset, output_path=output_path, mode="r+", compute=True)

    return xr_dataset


def create_empty_like(
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    keep_coordinates: bool = False,
    overwrite: bool = False,
) -> xr_Dataset:
    """A function that creates an empty MDIO v1 file with the same structure as an existing one.

    Args:
        input_path: The path of the input MDIO file.
        output_path: The path of the output MDIO file.
                     If None, the output will not be written to disk.
        keep_coordinates: Whether to keep the coordinates in the output file.
        overwrite: Whether to overwrite the output file if it exists.

    Returns:
        The output MDIO dataset.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
    """
    input_path = _normalize_path(input_path)
    output_path = _normalize_path(output_path) if output_path is not None else None

    if not overwrite and output_path is not None and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    ds = open_mdio(input_path)

    # Create a copy with the same structure but no data or,
    # optionally, coordinates
    ds_output = ds.copy(data=None).reset_coords(drop=not keep_coordinates)

    # Dataset
    # Keep the name (which is the same as the used template name) and the original API version
    # ds_output.attrs["name"]
    # ds_output.attrs["apiVersion"]
    ds_output.attrs["createdOn"] = str(datetime.now(UTC))

    # Coordinates
    if not keep_coordinates:
        for coord_name in ds_output.coords:
            ds_output[coord_name].attrs.pop("unitsV1", None)

    # MDIO attributes
    attr = ds_output.attrs["attributes"]
    if attr is not None:
        attr.pop("gridOverrides", None)  # Empty dataset should not have gridOverrides
        # Keep the original values for the following attributes
        # attr["defaultVariableName"]
        # attr["surveyType"]
        # attr["gatherType"]

    # "All traces should be marked as dead in empty dataset"
    if "trace_mask" in ds_output.variables:
        ds_output["trace_mask"][:] = False

    # Data variable
    var_name = attr["defaultVariableName"]
    var = ds_output[var_name]
    var.attrs.pop("statsV1", None)
    if not keep_coordinates:
        var.attrs.pop("unitsV1", None)

    # SEG-Y file header
    if "segy_file_header" in ds_output.variables:
        segy_file_header = ds_output["segy_file_header"]
        if segy_file_header is not None:
            segy_file_header.attrs.pop("textHeader", None)
            segy_file_header.attrs.pop("binaryHeader", None)
            segy_file_header.attrs.pop("rawBinaryHeader", None)

    if output_path is not None:
        to_mdio(ds_output, output_path=output_path, mode="w", compute=True)

    return ds_output
