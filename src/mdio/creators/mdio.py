"""Creating MDIO v1 datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdio.api.io import _normalize_path
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
    from mdio.core.dimension import Dimension


def create_empty(  # noqa PLR0913
    mdio_template_name: str,
    dimensions: list[Dimension],
    output_path: UPath | Path | str,
    headers: HeaderSpec | None = None,
    overwrite: bool = False,
) -> None:
    """A function that creates an empty MDIO v1 file with known dimensions.

    Args:
        mdio_template_name: The MDIO template to use to define the dataset structure.
        dimensions: The dimensions of the MDIO file.
        output_path: The universal path for the output MDIO v1 file.
        headers: SEG-Y v1.0 trace headers. Defaults to None.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
    """
    output_path = _normalize_path(output_path)

    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    header_dtype = to_structured_type(headers.dtype) if headers else None
    grid = Grid(dims=dimensions)
    mdio_template = TemplateRegistry().get(mdio_template_name)
    mdio_ds: Dataset = mdio_template.build_dataset(
        name=mdio_template_name,
        sizes=grid.shape,
        horizontal_coord_unit=None,
        header_dtype=header_dtype,
    )

    # Convert to xarray dataset
    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # Populate coordinates using the grid
    # For empty datasets, we only populate dimension coordinates
    drop_vars_delayed = []
    dataset, drop_vars_delayed = populate_dim_coordinates(xr_dataset, grid, drop_vars_delayed=drop_vars_delayed)

    if headers:
        # Since the headers were provided, the user wants to export to SEG-Y
        # Add a dummy segy_file_header variable used to export to SEG-Y
        dataset["segy_file_header"] = ((), "")

    # Create the Zarr store with the correct structure but with empty arrays
    to_mdio(dataset, output_path=output_path, mode="w", compute=False)

    # Write the dimension coordinates and trace mask
    meta_ds = dataset[drop_vars_delayed + ["trace_mask"]]
    to_mdio(meta_ds, output_path=output_path, mode="r+", compute=True)
