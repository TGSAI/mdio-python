"""SEG-Y creation utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
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


def mdio_spec_to_segy(  # noqa: PLR0913 DOC107
    mdio_path_or_buffer: str,
    output_segy_path: str,
    access_pattern: str,
    output_endian: Endianness,
    storage_options: dict[str],
    new_chunks: tuple[int, ...],
    backend: str,
) -> tuple[MDIOReader, SegyFactory]:
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


@dataclass(slots=True)
class SegyPartRecord:
    """Dataclass that holds partial SEG-Y record path and its global index."""

    path: str
    index: tuple[int, ...]


def serialize_to_segy_stack(
    samples: NDArray,
    headers: NDArray,
    live_mask: NDArray,
    record_ndim: int,
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
        record_ndim: First array dimensions to partition the SEGY record.
        file_root: Root directory to write partial SEG-Y files.
        segy_factory: A SEG-Y factory configured to write out with user params.
        block_info: Dask map_blocks reserved kwarg for block indices / shape etc.

    Returns:
        Live mask, as is, for combined blocks (dropped sample dimension).

    Raises:
        ValueError: If required `block_info` is not provided.
    """
    if block_info is None:
        raise ValueError("block_info is required for global index computation.")

    # Drop map_blocks padded dim
    live_mask = live_mask[..., 0]
    headers = headers[..., 0]

    # Figure out global chunk origin and shape of chunk
    info = block_info[0]
    block_start = [loc[0] for loc in info["array-location"]]

    if samples.ndim == 2:  # 2D data special case for less disk I/O
        # Shortcut if whole chunk is empty
        if np.count_nonzero(live_mask) == 0:
            return np.array(0, dtype=object)

        samples = samples[live_mask]
        headers = headers[live_mask]

        buffer = segy_factory.create_traces(headers, samples)

        global_index = block_start[0]
        record_id_str = str(global_index)
        record_file_path = f"{file_root}/{record_id_str}.bin"
        os.makedirs(os.path.dirname(record_file_path), exist_ok=True)
        with open(record_file_path, mode="wb") as fp:
            fp.write(buffer)

        record_metadata = SegyPartRecord(
            path=record_file_path,
            index=global_index,
        )
        records_metadata = np.array(record_metadata, dtype=object)

    else:  # 3D+ case where we unwrap first `record_ndim` axes.
        record_shape = samples.shape[:record_ndim]
        records_metadata = np.zeros(shape=record_shape, dtype=object)

        # Shortcut if whole chunk is empty
        if np.count_nonzero(live_mask) == 0:
            return records_metadata

        for rec_index in np.ndindex(record_shape):
            rec_live_mask = live_mask[rec_index]

            if np.count_nonzero(rec_live_mask) == 0:
                continue

            rec_samples = samples[rec_index][rec_live_mask]
            rec_headers = headers[rec_index][rec_live_mask]

            buffer = segy_factory.create_traces(rec_headers, rec_samples)

            global_index = tuple(
                block_start[i] + rec_index[i] for i in range(record_ndim)
            )
            record_id_str = "/".join(map(str, global_index))
            record_file_path = f"{file_root}/{record_id_str}.bin"
            os.makedirs(os.path.dirname(record_file_path), exist_ok=True)
            with open(record_file_path, mode="wb") as fp:
                fp.write(buffer)

            records_metadata[rec_index] = SegyPartRecord(
                path=record_file_path,
                index=global_index,
            )

    return records_metadata


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
