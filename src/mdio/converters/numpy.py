"""Conversion from Numpy to MDIO v1 format."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from mdio.api.io import _normalize_path
from mdio.api.io import to_mdio
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.converters.type_converter import to_scalar_type
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.core.dimension import Dimension
from mdio.core.grid import Grid
from mdio.core.utils_write import MAX_COORDINATES_BYTES
from mdio.core.utils_write import MAX_SIZE_LIVE_MASK
from mdio.core.utils_write import get_constrained_chunksize

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import DTypeLike
    from numpy.typing import NDArray
    from upath import UPath
    from xarray import Dataset as xr_Dataset

    from mdio.builder.schemas import Dataset

logger = logging.getLogger(__name__)


def get_compressor(lossless: bool, compression_tolerance: float) -> list:
    """Get compressor configuration based on compression settings."""
    if lossless:
        return [Blosc(cname=BloscCname.zstd)]
    else:
        # For lossy compression, we would need ZFP, but let's keep it simple for now
        # and use lossless as fallback
        logger.warning("Lossy compression not yet implemented, using lossless")
        return [Blosc(cname=BloscCname.zstd)]


def _prepare_inputs(
    array: NDArray,
    mdio_template: AbstractDatasetTemplate,
    chunksize: tuple[int, ...] | None,
    index_coords: dict[str, NDArray] | None,
) -> tuple[tuple[int, ...], dict[str, NDArray]]:
    """Prepare inputs and set defaults for chunksize and coordinates."""
    dim_names = mdio_template.dimension_names

    # Use template's chunk shape if not provided
    if chunksize is None:
        chunksize = mdio_template.full_chunk_shape

    # Create default coordinates if not provided
    if index_coords is None:
        index_coords = {}
        for name, size in zip(dim_names, array.shape, strict=True):
            index_coords[name] = np.arange(size, dtype=np.int32)

    return chunksize, index_coords


def _build_grid_and_dataset(
    array: NDArray,
    mdio_template: AbstractDatasetTemplate,
    chunksize: tuple[int, ...],
    index_coords: dict[str, NDArray],
    lossless: bool,
    compression_tolerance: float,
    header_dtype: DTypeLike | None,
) -> tuple[Grid, Dataset]:
    """Build the grid and dataset for the numpy array using the provided template."""
    # Create dimensions
    dims = [Dimension(name=name, coords=index_coords[name]) for name in mdio_template.dimension_names]
    grid = Grid(dims=dims)

    # Get compressor
    compressors = get_compressor(lossless, compression_tolerance)

    # Convert numpy dtype to MDIO ScalarType
    data_type = to_scalar_type(array.dtype)

    # Build dataset
    mdio_ds: Dataset = mdio_template.build_dataset(
        name=mdio_template.name,
        sizes=array.shape,
        header_dtype=header_dtype,
    )

    # Update the default variable with correct dtype and compressor
    var_index = next((i for i, v in enumerate(mdio_ds.variables) if v.name == mdio_template.default_variable_name), None)
    if var_index is not None:
        mdio_ds.variables[var_index].data_type = data_type
        mdio_ds.variables[var_index].compressor = compressors[0]

        # Set chunk grid for the data variable
        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=chunksize))
        if mdio_ds.variables[var_index].metadata is None:
            mdio_ds.variables[var_index].metadata = VariableMetadata()
        mdio_ds.variables[var_index].metadata.chunk_grid = chunk_grid

    # Dynamically chunk the trace_mask
    _chunk_variable(ds=mdio_ds, target_variable_name="trace_mask")

    # Dynamically chunk coordinate variables
    for coord in mdio_template.coordinate_names:
        _chunk_variable(ds=mdio_ds, target_variable_name=coord)

    return grid, mdio_ds


def _chunk_variable(ds: Dataset, target_variable_name: str) -> None:
    """Determines and sets the chunking for a specific Variable in the Dataset."""
    # Find variable index by name
    index = next((i for i, obj in enumerate(ds.variables) if obj.name == target_variable_name), None)
    if index is None:
        return

    def determine_target_size(var_type: str) -> int:
        """Determines the target size (in bytes) for a Variable based on its type."""
        if var_type == "bool":
            return MAX_SIZE_LIVE_MASK
        return MAX_COORDINATES_BYTES

    # Create the chunk grid metadata
    var_type = ds.variables[index].data_type
    full_shape = tuple(dim.size for dim in ds.variables[index].dimensions)
    target_size = determine_target_size(var_type)

    chunk_shape = get_constrained_chunksize(full_shape, var_type, target_size)
    chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=chunk_shape))

    # Create variable metadata if it doesn't exist
    if ds.variables[index].metadata is None:
        ds.variables[index].metadata = VariableMetadata()

    ds.variables[index].metadata.chunk_grid = chunk_grid


def _populate_coordinates_and_write(
    xr_dataset: xr_Dataset,
    grid: Grid,
    output_path: UPath | Path,
    array: NDArray,
) -> None:
    """Populate coordinates and write data to MDIO."""
    # Populate dimension coordinates
    drop_vars_delayed = []
    for dim in grid.dims:
        xr_dataset[dim.name].values[:] = dim.coords
        drop_vars_delayed.append(dim.name)

    # Set trace mask to all True (no missing data for numpy arrays)
    xr_dataset.trace_mask.data[:] = True
    drop_vars_delayed.append("trace_mask")

    # Create data dataset with the numpy array
    data_var_name = xr_dataset.attrs.get("defaultVariableName", "amplitude")
    data_ds = xr_dataset[[data_var_name]].copy()
    data_ds[data_var_name].data[:] = array

    # Combine all datasets
    full_ds = xr_dataset[drop_vars_delayed].merge(data_ds)

    # Write everything at once
    to_mdio(full_ds, output_path=output_path, mode="w", compute=True)


def numpy_to_mdio(  # noqa: PLR0913
    array: NDArray,
    mdio_template: AbstractDatasetTemplate,
    output_path: UPath | Path | str,
    chunksize: tuple[int, ...] | None = None,
    index_coords: dict[str, NDArray] | None = None,
    header_dtype: DTypeLike | None = None,
    lossless: bool = True,
    compression_tolerance: float = 0.01,
    storage_options: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Convert a NumPy array to MDIO v1 format.

    This function converts a NumPy array into the MDIO format following the same
    interface pattern as SEG-Y to MDIO conversion.

    Args:
        array: Input NumPy array to be converted to MDIO format.
        mdio_template: The MDIO template to use for the conversion. The template defines
            the dataset structure, but coordinate information is ignored for NumPy arrays.
        output_path: The universal path for the output MDIO v1 file.
        chunksize: Tuple specifying the chunk sizes for each dimension of the array. If not
            provided, uses the template's default chunk shape. Must match the number of
            dimensions in the input array.
        index_coords: Dictionary mapping dimension names to their coordinate arrays. If not
            provided, defaults to sequential integers (0 to size-1) for each dimension.
        header_dtype: Data type for trace headers, if applicable. Defaults to None.
        lossless: If True, uses lossless Blosc compression with zstandard. If False, uses ZFP lossy
            compression (requires `zfpy` library).
        compression_tolerance: Tolerance for ZFP compression in lossy mode. Ignored if
            `lossless=True`. Default is 0.01, providing ~70% size reduction.
        storage_options: Dictionary of storage options for the MDIO output file (e.g.,
            cloud credentials). Defaults to None (anonymous access).
        overwrite: If True, overwrites existing MDIO file at the specified path.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.

    Examples:
        Convert a 3D NumPy array to MDIO format using a template:

        >>> import numpy as np
        >>> from mdio.converters import numpy_to_mdio
        >>> from mdio.builder.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
        >>>
        >>> array = np.random.rand(100, 200, 300)
        >>> template = Seismic3DPostStackTemplate(data_domain="time")
        >>> numpy_to_mdio(
        ...     array=array,
        ...     mdio_template=template,
        ...     output_path="output/file.mdio",
        ...     chunksize=(64, 64, 64),
        ... )

        For a cloud-based output on AWS S3 with custom coordinates:

        >>> coords = {
        ...     "inline": np.arange(0, 100, 2),
        ...     "crossline": np.arange(0, 200, 4),
        ...     "time": np.linspace(0, 0.3, 300),
        ... }
        >>> numpy_to_mdio(
        ...     array=array,
        ...     mdio_template=template,
        ...     output_path="s3://bucket/file.mdio",
        ...     chunksize=(32, 32, 128),
        ...     index_coords=coords,
        ...     lossless=False,
        ...     compression_tolerance=0.01,
        ... )

        Convert a 2D array with default indexing and lossless compression:

        >>> from mdio.builder.templates.seismic_2d_poststack import Seismic2DPostStackTemplate
        >>> array_2d = np.random.rand(500, 1000)
        >>> template_2d = Seismic2DPostStackTemplate(data_domain="time")
        >>> numpy_to_mdio(
        ...     array=array_2d,
        ...     mdio_template=template_2d,
        ...     output_path="output/file_2d.mdio",
        ...     chunksize=(512, 512),
        ... )
    """
    storage_options = storage_options or {}

    # Prepare inputs and set defaults
    chunksize, index_coords = _prepare_inputs(array, mdio_template, chunksize, index_coords)

    # Normalize path
    output_path = _normalize_path(output_path)

    # Check if output exists
    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    # Build grid and dataset
    grid, mdio_ds = _build_grid_and_dataset(
        array=array,
        mdio_template=mdio_template,
        chunksize=chunksize,
        index_coords=index_coords,
        lossless=lossless,
        compression_tolerance=compression_tolerance,
        header_dtype=header_dtype,
    )

    # Convert to xarray dataset
    xr_dataset: xr_Dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    # Populate coordinates and write data
    _populate_coordinates_and_write(
        xr_dataset=xr_dataset,
        grid=grid,
        output_path=output_path,
        array=array,
    )
