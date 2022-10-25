"""SEG-Y creation utilities."""


from __future__ import annotations

import os
from shutil import copyfileobj
from time import sleep

import numpy as np
import segyio
from numpy.typing import NDArray
from segyio.binfield import keys as bfkeys

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


def preprocess_samples(
    samples: NDArray,
    live: NDArray,
    out_dtype: Dtype,
    out_byteorder: ByteOrder,
) -> NDArray:
    """Pre-process samples before writing to SEG-Y.

    Args:
        samples: Array containing the trace samples.
        live: live mask for the `samples` array.
        out_dtype: Output format for SEG-Y.
        out_byteorder: Output byte-order.

    Returns:
        Array with values modified according to configuration.
    """
    if np.count_nonzero(live) == 0:
        return np.empty_like(samples)

    if out_dtype == Dtype.IBM32:
        out_samples = samples.astype("float32", copy=False)
        out_samples = ieee2ibm(out_samples)
    else:
        out_samples = samples.astype(out_dtype, copy=False)

    in_byteorder = get_byteorder(samples)

    if in_byteorder != out_byteorder:
        out_samples.byteswap(inplace=True)

    return out_samples


def preprocess_headers(
    headers: NDArray,
    live: NDArray,
    out_byteorder: ByteOrder,
) -> NDArray:
    """Pre-process samples before writing to SEG-Y.

    Args:
        headers: Array containing the trace samples.
        live: live mask for the `headers` array.
        out_byteorder: Output byte-order.

    Returns:
        Array with values modified according to configuration.
    """
    out_headers = headers

    if np.count_nonzero(live) == 0:
        return out_headers

    in_byteorder = get_byteorder(headers)

    if in_byteorder != out_byteorder:
        out_headers.byteswap(inplace=True)

    return out_headers


# TODO: Abstract this to support various implementations by
#  object stores and file systems. Probably fsspec solution.
def concat_files(paths: list[str]) -> str:
    """Concatenate files on disk, sequentially in given order.

    This function takes files on disk, and it combines them by
    concatenation. Input files are deleted after merge, so disk
    usage doesn't explode.

    This is only required for disk / on-prem; since object stores
    have their optimized file concatenation implementations.

    Args:
        paths: Paths to the blocks of SEG-Y.

    Returns:
        Path to the returned file (first one from input).
    """
    first_file = paths.pop(0)

    for next_file in paths:
        with open(first_file, "ab+") as first_fp:
            with open(next_file, "rb") as next_fp:
                copyfileobj(next_fp, first_fp)

            os.remove(next_file)

    return first_file
