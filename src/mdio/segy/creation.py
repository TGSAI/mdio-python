"""SEG-Y creation utilities."""


from __future__ import annotations

import os
from os import path
from shutil import copyfileobj
from time import sleep
from uuid import uuid4

import numpy as np
import segyio
from numpy.typing import NDArray
from segyio.binfield import keys as bfkeys
from tqdm.auto import tqdm

from mdio.api.accessor import MDIOReader
from mdio.segy._standards_common import SegyFloatFormat
from mdio.segy.byte_utils import ByteOrder
from mdio.segy.byte_utils import Dtype
from mdio.segy.byte_utils import get_byteorder
from mdio.segy.ibm_float import ieee2ibm


def mdio_spec_to_segy(
    mdio_path_or_buffer,
    output_segy_path,
    endian,
    access_pattern,
    out_sample_format,
    storage_options,
    new_chunks,
    selection_mask,
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
    tracecount = mdio.trace_count  # Only LIVE
    if selection_mask is not None:
        tracecount = (grid.live_mask & selection_mask).sum()
    sample_dim = grid.select_dim("sample")  # Last axis is samples

    # Convert text header to bytearray. First merge all lines, and then encode.
    text_header = "".join(mdio.text_header)
    text_header = text_header.encode()

    # Get binary header dictionary
    binary_header = mdio.binary_header

    # Check and set output sample format.
    # This is only executed if the user parameter
    # doesn't match the original SEG-Y that was ingested.
    out_sample_format = SegyFloatFormat[out_sample_format.upper()]
    if out_sample_format != binary_header["Format"]:
        binary_header["Format"] = out_sample_format

    # Make sure binary header for format is in sync
    sample_format = binary_header["Format"]

    # Use `segyio` to create file and write initial metadata
    segy_spec = segyio.spec()
    segy_spec.samples = sample_dim
    segy_spec.endian = endian
    segy_spec.tracecount = tracecount
    segy_spec.format = sample_format

    with segyio.create(output_segy_path, segy_spec) as output_segy_file:
        # Write text and binary headers.
        # For binary header, we use the key mappings (str -> byte loc) from segyio
        output_segy_file.text[0] = text_header
        output_segy_file.bin = [
            (bfkeys[key], binary_header[key]) for key in binary_header
        ]

    return mdio, out_sample_format


def write_to_segy_stack(
    samples: NDArray,
    headers: NDArray,
    live: NDArray,
    out_dtype: Dtype,
    out_byteorder: ByteOrder,
    file_root: str,
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

    samples = cast_sample_format(samples, out_dtype)
    samples = check_byteswap(samples, out_byteorder)
    headers = check_byteswap(headers, out_byteorder)

    trace_dtype = np.dtype(
        {
            "names": ("header", "pad", "trace"),
            "formats": [
                headers.dtype,
                np.dtype("int64"),
                samples.shape[-1] * samples.dtype,
            ],
        },
    )

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

        # Interleave samples and headers
        part_traces = np.empty(num_live, dtype=trace_dtype)
        part_traces["header"] = headers[index][part_live]
        part_traces["trace"] = samples[index][part_live]
        part_traces["pad"] = 0

        with open(file_path, mode="wb") as fp:
            part_traces.tofile(fp)

    return part_segy_paths


def check_byteswap(array: NDArray, out_byteorder: ByteOrder) -> NDArray:
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


def cast_sample_format(
    samples: NDArray,
    out_dtype: Dtype,
) -> NDArray:
    """Cast sample format (dtype).

    Args:
        samples: Array containing the trace samples.
        out_dtype: Desired output data type.

    Returns:
        New structured array with pre-processing applied.
    """
    if out_dtype == Dtype.IBM32:
        samples = samples.astype("float32", copy=False)
        samples = ieee2ibm(samples)
    else:
        samples = samples.astype(out_dtype, copy=False)

    return samples


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
