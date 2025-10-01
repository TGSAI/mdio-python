"""Creating MDIO v1 datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from segy.standards import get_segy_standard

from mdio.api.io import _normalize_path
from mdio.api.io import to_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.converters.segy import get_horizontal_coordinate_unit
from mdio.converters.segy import populate_dim_coordinates
from mdio.converters.type_converter import to_structured_type
from mdio.core.grid import Grid

if TYPE_CHECKING:
    from pathlib import Path

    from upath import UPath
    from xarray import Dataset as xr_Dataset

    from mdio.builder.schemas import Dataset
    from mdio.core.dimension import Dimension


def create_empty_mdio(  # noqa PLR0913
    mdio_template_name: str,
    dimensions: list[Dimension],
    output_path: UPath | Path | str,
    create_headers: bool = False,
    overwrite: bool = False,
) -> None:
    """A function that creates an empty MDIO v1 file with known dimensions.

    Args:
        mdio_template_name: The MDIO template to use to define the dataset structure.
        dimensions: The dimensions of the MDIO file.
        output_path: The universal path for the output MDIO v1 file.
        create_headers: Whether to create a full set of SEG-Y v1.0 trace headers. Defaults to False.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
    """
    output_path = _normalize_path(output_path)

    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    header_dtype = to_structured_type(get_segy_standard(1.0).trace.header.dtype) if create_headers else None
    grid = Grid(dims=dimensions)
    horizontal_unit = get_horizontal_coordinate_unit(grid.dims)
    mdio_template = TemplateRegistry().get(mdio_template_name)
    mdio_ds: Dataset = mdio_template.build_dataset(
        name=mdio_template_name,
        sizes=grid.shape,
        horizontal_coord_unit=horizontal_unit,
        header_dtype=header_dtype,
    )

    # Convert to xarray dataset
    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # Populate coordinates using the grid
    # For empty datasets, we only populate dimension coordinates
    drop_vars_delayed = []
    dataset, drop_vars_delayed = populate_dim_coordinates(xr_dataset, grid, drop_vars_delayed=drop_vars_delayed)

    # Set the trace mask to indicate all traces are live (since this is an empty dataset)
    dataset.trace_mask.data[:] = True

    # Create the Zarr store with the correct structure but with empty arrays
    to_mdio(dataset, output_path=output_path, mode="w", compute=False)

    # Write the dimension coordinates and trace mask
    meta_ds = dataset[drop_vars_delayed + ["trace_mask"]]
    to_mdio(meta_ds, output_path=output_path, mode="r+", compute=True)
