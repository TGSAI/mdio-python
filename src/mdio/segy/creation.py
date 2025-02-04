"""SEG-Y creation utilities."""

from __future__ import annotations

import logging
import os
from shutil import copyfileobj
from typing import TYPE_CHECKING

import numpy as np
from segy.factory import SegyFactory
from segy.schema import Endianness
from segy.schema import SegySpec
from tqdm.auto import tqdm

from mdio.api.accessor import MDIOReader
from mdio.segy.compat import mdio_segy_spec
from mdio.segy.compat import revision_encode
from mdio.segy.utilities import find_trailing_ones_index
from mdio.segy.utilities import ndrange


if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def make_segy_factory(
    mdio: MDIOReader,
    spec: SegySpec,
) -> SegyFactory:
    """Generate SEG-Y factory from MDIO metadata."""
    grid = mdio.grid
    sample_dim = grid.select_dim("sample")
    sample_interval = sample_dim[1] - sample_dim[0]
    samples_per_trace = len(sample_dim)

    return SegyFactory(
        spec=spec,
        sample_interval=sample_interval * 1000,
        samples_per_trace=samples_per_trace,
    )


def mdio_spec_to_segy(
    mdio_path_or_buffer,
    output_segy_path,
    access_pattern,
    output_endian,
    storage_options,
    new_chunks,
    backend,
):
    """Create SEG-Y file without any traces given MDIO specification.

    This function opens an MDIO file, gets some relevant information for SEG-Y files,
    then creates a SEG-Y file with the specification it read from the MDIO file.

    It then returns the `MDIOReader` instance, and the parsed floating point format
    `sample_format` for further use.

    Function will attempt to read text, and binary headers, and some grid information
    from the MDIO file. If these don't exist, the process will fail.

    Args:
        mdio_path_or_buffer: Input path where the MDIO is located.
        output_segy_path: Path to the output SEG-Y file.
        access_pattern: This specificies the chunk access pattern.
            Underlying zarr.Array must exist. Examples: '012', '01'.
        output_endian: Endianness of the output file.
        storage_options: Storage options for the cloud storage backend.
            Default: None (will assume anonymous)
        new_chunks: Set manual chunksize. For development purposes only.
        backend: Eager (zarr) or lazy but more flexible 'dask' backend.

    Returns:
        Initialized MDIOReader for MDIO file and return SegyFactory
    """
    mdio = MDIOReader(
        mdio_path_or_buffer=mdio_path_or_buffer,
        access_pattern=access_pattern,
        storage_options=storage_options,
        return_metadata=True,
        new_chunks=new_chunks,
        backend=backend,
        memory_cache_size=0,  # Making sure disk caching is disabled
        disk_cache=False,  # Making sure disk caching is disabled
    )

    mdio_file_version = mdio.root.attrs["api_version"]
    spec = mdio_segy_spec(mdio_file_version)
    spec.endianness = Endianness(output_endian)
    factory = make_segy_factory(mdio, spec=spec)

    text_str = "\n".join(mdio.text_header)
    text_bytes = factory.create_textual_header(text_str)

    binary_header = revision_encode(mdio.binary_header, mdio_file_version)
    bin_hdr_bytes = factory.create_binary_header(binary_header)

    with open(output_segy_path, mode="wb") as fp:
        fp.write(text_bytes)
        fp.write(bin_hdr_bytes)

    return mdio, factory


def serialize_to_segy_stack(
    samples: NDArray,
    headers: NDArray,
    live_mask: NDArray,
    file_root: str,
    segy_factory: SegyFactory,
    block_info: dict | None = None,
) -> NDArray:
    """Pre-process seismic data for SEG-Y and write partial 2D blocks.

    This function will take numpy arrays for trace samples, headers, and live mask.
    Then it will do the following:
    1. Iterate outer dimensions that are wrapped.
    2. Drop non-live samples and headers.
    3. Combine samples and headers to form a SEG-Y trace.
    4. Write serialized bytes to disk.

    Args:
        samples: Array containing the trace samples.
        headers: Array containing the trace headers.
        live_mask: Array containing the trace live mask.
        file_root: Root directory to write partial SEG-Y files.
        segy_factory: A SEG-Y factory configured to write out with user params.
        block_info: Dask map_blocks reserved kwarg for block indices / shape etc.

    Returns:
        Live mask, as is, for combined blocks (dropped sample dimension).
    """
    # Drop map_blocks padded dim
    live_mask = live_mask[..., 0]
    headers = headers[..., 0]

    if block_info is None:
        return live_mask

    if np.count_nonzero(live_mask) == 0:
        return live_mask

    # Set up chunk boundaries and coordinates to write
    global_num_blocks = block_info[0]["num-chunks"]
    block_coords = block_info[0]["array-location"]
    result_chunk_shape = block_info[None]["chunk-shape"]

    # Find dimensions that are not chunked to -1 (full size)
    # Typically outer (slow changing) dimensions
    consecutive_dim_index = find_trailing_ones_index(global_num_blocks)
    prefix_block_coords = block_coords[:consecutive_dim_index]
    prefix_block_shape = result_chunk_shape[:consecutive_dim_index]

    # Generate iterators for dimension's index and coords
    indices_iter = np.ndindex(prefix_block_shape)
    coords_iter = ndrange(prefix_block_coords)

    # This pulls 2D unchunked slices out of the ND chunks
    # Writes them to global coordinates so we can combine them
    # in the right order later
    for dim_indices, dim_coords in zip(indices_iter, coords_iter, strict=True):
        # TODO(Altay): When python minimum is 3.11 change to live_mask[*dim_indices]
        aligned_live_mask = live_mask[tuple(dim_indices)]

        if np.count_nonzero(aligned_live_mask) == 0:
            continue

        # TODO(Altay): When python minimum is 3.11 change to samples[*dim_indices]
        aligned_samples = samples[tuple(dim_indices)][aligned_live_mask]
        aligned_headers = headers[tuple(dim_indices)][aligned_live_mask]

        buffer = segy_factory.create_traces(aligned_headers, aligned_samples)

        aligned_filename = "/".join(map(str, dim_coords))
        aligned_path = f"{file_root}/{aligned_filename}._mdiotemp"
        os.makedirs(os.path.dirname(aligned_path), exist_ok=True)
        with open(aligned_path, mode="wb") as fp:
            fp.write(buffer)

    return live_mask


# TODO: Abstract this to support various implementations by
#  object stores and file systems. Probably fsspec solution.
def concat_files(paths: list[str], progress=False) -> str:
    """Concatenate files on disk, sequentially in given order.

    This function takes files on disk, and it combines them by
    concatenation. Input files are deleted after merge, so disk
    usage doesn't explode.

    This is only required for disk / on-prem; since object stores
    have their optimized file concatenation implementations.

    Args:
        paths: Paths to the blocks of SEG-Y.
        progress: Enable tqdm progress bar. Default is False.

    Returns:
        Path to the returned file (first one from input).
    """
    first_file = paths.pop(0)

    if progress is True:
        paths = tqdm(paths, desc="Merging lines")

    with open(first_file, "ab+") as first_fp:
        for next_file in paths:
            with open(next_file, "rb") as next_fp:
                copyfileobj(next_fp, first_fp)

            os.remove(next_file)

    return first_file
