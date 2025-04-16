"""Module for creating empty MDIO datasets.

This module provides tools to configure and initialize empty MDIO datasets, which are
used for storing multidimensional data with associated metadata. It includes:

- `MDIOVariableConfig`: Config for individual variables in the dataset.
- `MDIOCreateConfig`: Config for the dataset, including path, grid, and variables.
- `create_empty`: Function to create the empty dataset based on provided configuration.
- `create_empty_like`: Create an empty dataset with same structure as an existing one.

The `create_empty` function sets up the Zarr hierarchy with metadata and data groups,
creates datasets for each variable and their trace headers, and initializes attributes
such as creation time, API version, grid dimensions, and basic statistics.

The `create_empty_like` function creates a new empty dataset by replicating the
structure of an existing MDIO dataset, including its grid, variables, and headers.

For detailed usage and parameters, see the docstring of the `create_empty` function.
"""

from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from importlib import metadata
from typing import Any

import zarr
from numcodecs import Blosc
from numpy.typing import DTypeLike
from zarr import Group
from zarr import open_group
from zarr.core.array import CompressorsLike

from mdio.api.accessor import MDIOWriter
from mdio.api.io_utils import process_url
from mdio.core.grid import Grid
from mdio.core.utils_write import get_live_mask_chunksize
from mdio.core.utils_write import write_attribute
from mdio.segy.compat import mdio_segy_spec
from mdio.segy.helpers_segy import create_zarr_hierarchy


try:
    API_VERSION = metadata.version("multidimio")
except metadata.PackageNotFoundError:
    API_VERSION = "unknown"

DEFAULT_TEXT = [f"C{idx:02d}" + " " * 77 for idx in range(40)]
DEFAULT_TRACE_HEADER_DTYPE = mdio_segy_spec().trace.header.dtype


@dataclass
class MDIOVariableConfig:
    """Configuration for creating an MDIO variable.

    This dataclass defines the parameters required to configure a variable
    in an MDIO dataset, including its name, data type, chunking strategy,
    compression method, and optional header data type.

    Attributes:
        name: The name of the variable.
        dtype: The data type of the variable (e.g., 'float32', 'int16').
        chunks: The chunk size for the variable along each dimension.
        compressors: The compression algorithm(s) to use.
        header_dtype: The data type for the variable's header.
    """

    name: str
    dtype: str
    chunks: tuple[int, ...] | None = None
    compressors: CompressorsLike | None = None
    header_dtype: DTypeLike | None = None


@dataclass
class MDIOCreateConfig:
    """Configuration for creating an MDIO dataset.

    This dataclass encapsulates the parameters needed to create an MDIO dataset,
    including the storage path, grid specification, and a list of variable
    configurations.

    Attributes:
        path: The file path or URI where the MDIO dataset will be created.
        grid: The grid specification defining the dataset's spatial structure.
        variables: A list of configurations for variables to be included in dataset.
    """

    path: str
    grid: Grid
    variables: list[MDIOVariableConfig]


def create_empty(
    config: MDIOCreateConfig,
    overwrite: bool = False,
    storage_options: dict[str, Any] | None = None,
    consolidate_meta: bool = True,
) -> Group:
    """Create an empty MDIO dataset.

    This function initializes a new MDIO dataset at the specified path based on the provided
    configuration. It constructs a Zarr hierarchy with groups for metadata and data, creates
    datasets for each variable and its associated trace headers, and sets initial attributes
    such as creation time, API version, grid dimensions, and basic statistics (all initialized
    to zero). An empty 'live_mask' dataset is also created to track valid traces.

    Important: It is up to the user to update live_mask and other attributes.

    Args:
        config: Configuration object specifying the dataset's path, grid structure, and a
            list of variable configurations (e.g., name, dtype, chunks).
        overwrite: If True, overwrites any existing dataset at the specified path. If
            False, an error is raised if the dataset exists. Defaults to False.
        storage_options: Options for the storage backend, such as credentials or settings for
            cloud storage (e.g., S3, GCS). Defaults to None.
        consolidate_meta: If True, consolidates metadata into a single file after creation,
            improving performance for large metadata. Defaults to True.

    Returns:
        Group: The root Zarr group representing the newly created MDIO dataset.
    """
    zarr.config.set({"default_zarr_format": 2, "write_empty_chunks": False})

    storage_options = storage_options or {}

    url = process_url(url=config.path, disk_cache=False)
    root_group = open_group(url, mode="w", storage_options=storage_options)
    root_group = create_zarr_hierarchy(root_group, overwrite)

    meta_group = root_group["metadata"]
    data_group = root_group["data"]

    # Get UTC time, then add local timezone information offset.
    iso_datetime = datetime.now(timezone.utc).isoformat()
    dimensions_dict = [dim.to_dict() for dim in config.grid.dims]

    write_attribute(name="created", zarr_group=root_group, attribute=iso_datetime)
    write_attribute(name="api_version", zarr_group=root_group, attribute=API_VERSION)
    write_attribute(name="dimension", zarr_group=root_group, attribute=dimensions_dict)
    write_attribute(name="trace_count", zarr_group=root_group, attribute=0)
    write_attribute(name="text_header", zarr_group=meta_group, attribute=DEFAULT_TEXT)
    write_attribute(name="binary_header", zarr_group=meta_group, attribute={})

    live_shape = config.grid.shape[:-1]
    live_chunks = get_live_mask_chunksize(live_shape)
    meta_group.create_array(
        name="live_mask",
        shape=live_shape,
        chunks=live_chunks,
        dtype="bool",
        chunk_key_encoding={"name": "v2", "separator": "/"},
    )

    for variable in config.variables:
        data_group.create_array(
            name=variable.name,
            shape=config.grid.shape,
            dtype=variable.dtype,
            compressors=variable.compressors,
            chunks=variable.chunks,
            chunk_key_encoding={"name": "v2", "separator": "/"},
        )

        header_dtype = variable.header_dtype or DEFAULT_TRACE_HEADER_DTYPE
        meta_group.create_array(
            name=f"{variable.name}_trace_headers",
            shape=config.grid.shape[:-1],  # Same spatial shape as data
            chunks=variable.chunks[:-1],  # Same spatial chunks as data
            compressors=Blosc("zstd"),
            dtype=header_dtype,
            chunk_key_encoding={"name": "v2", "separator": "/"},
        )

    stats = {
        "mean": 0,
        "std": 0,
        "rms": 0,
        "min": 0,
        "max": 0,
    }

    for key, value in stats.items():
        write_attribute(name=key, zarr_group=root_group, attribute=value)

    if consolidate_meta:
        zarr.consolidate_metadata(root_group.store)

    return root_group


def create_empty_like(
    source_path: str,
    dest_path: str,
    overwrite: bool = False,
    storage_options_input: dict[str, Any] | None = None,
    storage_options_output: dict[str, Any] | None = None,
) -> None:
    """Create an empty MDIO dataset with the same structure as an existing one.

    This function initializes a new empty MDIO dataset at the specified
    destination path, replicating the structure of an existing dataset, including
    its grid, variables, chunking strategy, compression, and headers. It copies
    metadata such as text and binary headers from the source dataset and sets
    initial attributes like creation time, API version, and zeroed statistics.

    Important: It is up to the user to update headers, `live_mask` and stats.

    Args:
        source_path: The path or URI of the existing MDIO dataset to replicate.
        dest_path: The path or URI where the new MDIO dataset will be created.
        overwrite: If True, overwrites any existing dataset at the destination.
        storage_options_input: Options for storage backend of the source dataset.
        storage_options_output: Options for storage backend of the destination dataset.
    """
    storage_options_input = storage_options_input or {}
    storage_options_output = storage_options_output or {}

    source_root = zarr.open_consolidated(
        source_path,
        mode="r",
        storage_options=storage_options_input,
    )
    src_data_grp = source_root["data"]
    src_meta_grp = source_root["metadata"]

    grid = Grid.from_zarr(source_root)

    variables = []
    for var_name in src_data_grp:
        variable = MDIOVariableConfig(
            name=var_name,
            dtype=src_data_grp[var_name].dtype,
            chunks=src_data_grp[var_name].chunks,
            compressors=src_data_grp[var_name].compressors,
            header_dtype=src_meta_grp[f"{var_name}_trace_headers"].dtype,
        )
        variables.append(variable)

    config = MDIOCreateConfig(path=dest_path, grid=grid, variables=variables)

    create_empty(
        config=config,
        overwrite=overwrite,
        storage_options=storage_options_output,
    )

    writer = MDIOWriter(dest_path, storage_options=storage_options_output)
    writer.text_header = src_meta_grp.attrs["text_header"]
    writer.binary_header = src_meta_grp.attrs["binary_header"]
