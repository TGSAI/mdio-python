"""Conversion from Numpy to MDIO v1 format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mdio.api.io import _normalize_path
from mdio.api.io import to_mdio
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.converters.type_converter import to_scalar_type

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import DTypeLike
    from numpy.typing import NDArray
    from upath import UPath
    from xarray import Dataset as xr_Dataset

    from mdio.builder.schemas import Dataset
    from mdio.builder.templates.base import AbstractDatasetTemplate


def _build_dataset(
    array: NDArray,
    mdio_template: AbstractDatasetTemplate,
    header_dtype: DTypeLike | None,
) -> Dataset:
    """Build the dataset for the numpy array using the provided template."""
    # Convert numpy dtype to MDIO ScalarType
    data_type = to_scalar_type(array.dtype)

    # Build dataset using template (which defines chunking)
    mdio_ds: Dataset = mdio_template.build_dataset(
        name=mdio_template.name,
        sizes=array.shape,
        header_dtype=header_dtype,
    )

    # Update the default variable with correct dtype
    var_index = next(
        (i for i, v in enumerate(mdio_ds.variables) if v.name == mdio_template.default_variable_name), None
    )
    if var_index is not None:
        mdio_ds.variables[var_index].data_type = data_type

    return mdio_ds


def _validate_coordinates(
    index_coords: dict[str, NDArray],
    mdio_template: AbstractDatasetTemplate,
    array: NDArray,
) -> None:
    """Validate user-provided coordinates against template and array dimensions.

    Args:
        index_coords: Dictionary mapping dimension names to coordinate arrays.
        mdio_template: The MDIO template defining expected dimensions.
        array: The numpy array being converted.

    Raises:
        ValueError: If coordinate names or sizes don't match requirements.
    """
    # Validate that coordinate names match template dimension names
    for coord_name in index_coords:
        if coord_name not in mdio_template.dimension_names:
            available_dims = sorted(mdio_template.dimension_names)
            err = (
                f"Coordinate name '{coord_name}' not found in template dimensions. "
                f"Available dimensions: {available_dims}"
            )
            raise ValueError(err)

    # Validate coordinate array sizes match array dimensions
    for dim_name, coord_array in index_coords.items():
        expected_size = array.shape[mdio_template.dimension_names.index(dim_name)]
        if coord_array.size != expected_size:
            err = (
                f"Size of coordinate '{dim_name}' ({coord_array.size}) does not match "
                f"array dimension size ({expected_size})"
            )
            raise ValueError(err)


def _populate_coordinates_and_write(
    xr_dataset: xr_Dataset,
    index_coords: dict[str, NDArray],
    output_path: UPath | Path,
    array: NDArray,
) -> None:
    """Populate coordinates and write data to MDIO."""
    # Populate dimension coordinates
    for name, coords in index_coords.items():
        xr_dataset[name].data[:] = coords

    # Set trace mask to all True (no missing data for numpy arrays)
    xr_dataset.trace_mask.data[:] = True

    # Set the data
    data_var_name = xr_dataset.attrs.get("defaultVariableName", "amplitude")
    xr_dataset[data_var_name].data[:] = array

    # Write everything at once
    to_mdio(xr_dataset, output_path=output_path, mode="w", compute=True)


def numpy_to_mdio(  # noqa: PLR0913
    array: NDArray,
    mdio_template: AbstractDatasetTemplate,
    output_path: UPath | Path | str,
    index_coords: dict[str, NDArray] | None = None,
    header_dtype: DTypeLike | None = None,
    overwrite: bool = False,
) -> None:
    """Convert a NumPy array to MDIO v1 format.

    This function converts a NumPy array into the MDIO format following the same
    interface pattern as SEG-Y to MDIO conversion.

    Args:
        array: Input NumPy array to be converted to MDIO format.
        mdio_template: The MDIO template to use for the conversion. The template defines
            the dataset structure including compression and chunking settings.
        output_path: The universal path for the output MDIO v1 file.
        index_coords: Dictionary mapping dimension names to their coordinate arrays. If not
            provided, defaults to sequential integers (0 to size-1) for each dimension.
        header_dtype: Data type for trace headers, if applicable. Defaults to None.
        overwrite: If True, overwrites existing MDIO file at the specified path.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
    """
    # Prepare coordinates - create defaults if not provided
    if index_coords is None:
        index_coords = {}
        for name, size in zip(mdio_template.dimension_names, array.shape, strict=True):
            index_coords[name] = np.arange(size, dtype=np.int32)

    # Normalize path
    output_path = _normalize_path(output_path)

    # Check if output exists
    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    # Validate coordinates if provided
    if index_coords:
        _validate_coordinates(index_coords, mdio_template, array)

    # Build dataset
    mdio_ds = _build_dataset(
        array=array,
        mdio_template=mdio_template,
        header_dtype=header_dtype,
    )

    # Convert to xarray dataset
    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # Populate coordinates and write data
    _populate_coordinates_and_write(
        xr_dataset=xr_dataset,
        index_coords=index_coords,
        output_path=output_path,
        array=array,
    )
