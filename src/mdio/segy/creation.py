"""SEG-Y creation utilities."""

from __future__ import annotations

import os
from os import path
from shutil import copyfileobj
from time import sleep
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from segy.factory import SegyFactory
from segy.schema.data_type import Endianness
from segy.schema.data_type import ScalarType
from segy.standards.rev0 import rev0_segy
from tqdm.auto import tqdm

from mdio.api.accessor import MDIOReader
from mdio.segy.byte_utils import get_byteorder


if TYPE_CHECKING:
    from numpy.typing import NDArray


def mdio_spec_to_segy(
    mdio_path_or_buffer,
    output_segy_path,
    endian,
    access_pattern,
    out_sample_format,
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
        endian: Endianness of the input SEG-Y. Rev.2 allows little endian.
            Default is 'big'. Must be in {"big", "little"}.
        access_pattern: This specificies the chunk access pattern.
            Underlying zarr.Array must exist. Examples: '012', '01'.
        out_sample_format: Output sample format. Currently support:
            {'ibm', 'ieee'}. Default is 'ibm'.
        storage_options: Storage options for the cloud storage backend.
            Default: None (will assume anonymous)
        new_chunks: Set manual chunksize. For development purposes only.
        selection_mask: Array that lists the subset of traces to be written
        backend: Eager (zarr) or lazy but more flexible 'dask' backend.

    Returns:
        Initialized MDIOReader for MDIO file and sample format parsed as integer
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

    sleep(0.5)  # So the connection message prints before tqdm

    # Get grid, tracecount, and sample dimension
    grid = mdio.grid
    sample_dim = grid.select_dim("sample")  # Last axis is samples

    # Get binary header dictionary
    binary_header = mdio.binary_header

    # Check and set output sample format.
    # This is only executed if the user parameter
    # doesn't match the original SEG-Y that was ingested.
    out_sample_format = ScalarType[out_sample_format.upper()]
    if out_sample_format != binary_header["Format"]:
        binary_header["Format"] = out_sample_format

    # Use `segy` to create file and write initial metadata
    segy_spec = rev0_segy
    segy_spec.endianness = Endianness[endian.upper()]
    sample_interval = sample_dim[1] - sample_dim[0]

    factory = SegyFactory(
        spec=segy_spec,
        sample_interval=sample_interval,
        samples_per_trace=len(sample_dim),
    )

    text_bytes = factory.create_textual_header(mdio.text_header)
    bin_hdr_bytes = factory.create_binary_header()

    with open(output_segy_path, mode="wb") as fp:
        # Write text and binary headers.
        # For binary header, we use the key mappings (str -> byte loc) from segyio
        fp.write(text_bytes)
        fp.write(bin_hdr_bytes)

    return mdio, out_sample_format


def write_to_segy_stack(
    samples: NDArray,
    headers: NDArray,
    live: NDArray,
    file_root: str,
    segy_factory: SegyFactory,
) -> NDArray:
    """Pre-process seismic data for SEG-Y and write partial blocks.

    This function will take numpy arrays for trace samples, headers, and live mask.
    Then it will do the following:
    1. Convert sample format to `out_dtype`.
    2. Byte-swap samples and headers if needed based on `out_byteorder`.
    3. Iterate inner dimensions and write blocks of traces.
    3.1. Combine samples and headers to form a SEG-Y trace.
    3.2. Drop non-live samples.
    3.3. Write line block to disk.
    3.4. Save file names for further merges
    4. Written files will be saved, so further merge methods
    can combine them to a single flat SEG-Y dataset.

    Args:
        samples: Array containing the trace samples.
        headers: Array containing the trace headers.
        live: Array containing the trace live mask.
        out_dtype: Desired output data type.
        out_byteorder: Desired output data byte order.
        file_root: Root directory to write partial SEG-Y files.

    Returns:
        Array containing file names for partial data. None means
        there were no live traces within the block / line.
    """
    # Make output array with string type. We need to know
    # the length of the string ahead of time.
    # Last axis can be written as sequential, so we collapse that to 1.
    mock_path = path.join(file_root, uuid4().hex)
    paths_dtype = f"U{len(mock_path)}"
    paths_shape = live.shape[:-1] + (1,)
    part_segy_paths = np.full(paths_shape, fill_value="missing", dtype=paths_dtype)

    # Fast path to return if no live traces.
    if np.count_nonzero(live) == 0:
        return part_segy_paths

    # Iterate on N-1 axes of live mask. Last axis can be written
    # without worrying about order because it is sequential.
    for index in np.ndindex(live.shape[:-1]):
        part_live = live[index]
        num_live = np.count_nonzero(part_live)

        if num_live == 0:
            continue

        # Generate unique file name and append to return list.
        file_path = path.join(file_root, uuid4().hex)
        part_segy_paths[index] = file_path

        # Create traces bytes
        trace_bytes = segy_factory.create_traces(
            headers=headers[index][part_live],
            samples=samples[index][part_live],
        )

        with open(file_path, mode="wb") as fp:
            fp.write(trace_bytes)

    return part_segy_paths


def check_byteswap(array: NDArray, out_byteorder: Endianness) -> NDArray:
    """Check input byteorder and swap if user wants the other.

    Args:
        array: Array containing the data.
        out_byteorder: Desired output data byte order.

    Returns:
        Original or byte-order swapped array.
    """
    in_byteorder = get_byteorder(array)

    if in_byteorder != out_byteorder:
        array.byteswap(inplace=True)
        array = array.newbyteorder()

    return array


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
