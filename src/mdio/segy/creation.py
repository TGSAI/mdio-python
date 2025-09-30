"""SEG-Y creation utilities."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfileobj
from typing import TYPE_CHECKING

import numpy as np
from segy.factory import SegyFactory
from segy.standards.fields import binary
from tqdm.auto import tqdm

from mdio.api.io import open_mdio
from mdio.exceptions import MDIOMissingVariableError
from mdio.segy.compat import encode_segy_revision

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray
    from segy.schema import SegySpec
    from upath import UPath


logger = logging.getLogger(__name__)


def make_segy_factory(spec: SegySpec, binary_header: dict[str, int]) -> SegyFactory:
    """Generate SEG-Y factory from MDIO metadata."""
    sample_interval = binary_header["sample_interval"]
    samples_per_trace = binary_header["samples_per_trace"]
    return SegyFactory(
        spec=spec,
        sample_interval=sample_interval,  # Sample interval is read in from binary header so no scaling here
        samples_per_trace=samples_per_trace,
    )


def mdio_spec_to_segy(
    segy_spec: SegySpec,
    input_path: UPath,
    output_path: UPath,
    new_chunks: tuple[int, ...] | None = None,
) -> tuple[xr.Dataset, SegyFactory]:
    """Create SEG-Y file without any traces given MDIO specification.

    This function opens an MDIO file, gets some relevant information for SEG-Y files, then creates
    a SEG-Y file with the specification it read from the MDIO file.

    It then returns the Xarray Dataset instance and SegyFactory for further use.

    Function will attempt to read text, and binary headers information from the MDIO file.
    If these don't exist, the process will fail.

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        input_path: Store or URL (and cloud options) for MDIO file.
        output_path: Path to the output SEG-Y file.
        new_chunks: Set in memory chunksize for export or other reasons.

    Returns:
        Opened Xarray Dataset for MDIO file and SegyFactory

    Raises:
        MDIOMissingVariableError: If MDIO file does not contain SEG-Y headers.
    """
    # NOTE: the warning analysis and the reason for its suppression are here:
    # https://github.com/TGSAI/mdio-python/issues/657
    warn = "The specified chunks separate the stored chunks along dimension"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=warn, category=UserWarning)
        dataset = open_mdio(input_path, chunks=new_chunks)

    if "segy_file_header" not in dataset:
        msg = (
            "MDIO does not contain SEG-Y file headers to write to output. Please add a dummy segy_file_header "
            "variable and fill its metadata (.attrs) with `textHeader` and `binaryHeader`."
        )
        raise MDIOMissingVariableError(msg)

    file_header = dataset["segy_file_header"]
    text_header = file_header.attrs["textHeader"]
    binary_header = file_header.attrs["binaryHeader"]
    binary_header = encode_segy_revision(binary_header)

    factory = make_segy_factory(spec=segy_spec, binary_header=binary_header)

    text_header_bytes = factory.create_textual_header(text_header)

    # During MDIO SEGY import, TGSAI/segy always creates revision major/minor fields
    # We may not have it in the user desired spec. In that case we add it here
    if "segy_revision" not in segy_spec.binary_header.names:
        rev_field = binary.Rev1.SEGY_REVISION.model
        segy_spec.binary_header.customize(fields=rev_field)

    binary_header_bytes = factory.create_binary_header(binary_header)

    with output_path.open(mode="wb") as fp:
        fp.write(text_header_bytes)
        fp.write(binary_header_bytes)

    return dataset, factory


@dataclass(slots=True)
class SegyPartRecord:
    """Dataclass that holds partial SEG-Y record path and its global index."""

    path: Path
    index: tuple[int, ...]


def serialize_to_segy_stack(  # noqa: PLR0913
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
        msg = "block_info is required for global index computation."
        raise ValueError(msg)

    # Drop map_blocks padded dim
    live_mask = live_mask[..., 0]
    headers = headers[..., 0]

    # Figure out global chunk origin and shape of chunk
    info = block_info[0]
    block_start = [loc[0] for loc in info["array-location"]]

    if samples.ndim == 2:  # noqa: PLR2004
        # 2D data special case for less disk I/O
        # Shortcut if whole chunk is empty
        if np.count_nonzero(live_mask) == 0:
            return np.array(0, dtype=object)

        samples = samples[live_mask]
        headers = headers[live_mask]

        buffer = segy_factory.create_traces(headers, samples)

        global_index = block_start[0]
        record_id_str = str(global_index)
        record_file_path = Path(file_root) / f"{record_id_str}.bin"
        record_file_path.parent.mkdir(parents=True, exist_ok=True)
        with record_file_path.open(mode="wb") as fp:
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

            global_index = tuple(block_start[i] + rec_index[i] for i in range(record_ndim))
            record_id_str = "/".join(map(str, global_index))
            record_file_path = Path(file_root) / f"{record_id_str}.bin"
            record_file_path.parent.mkdir(parents=True, exist_ok=True)
            with record_file_path.open(mode="wb") as fp:
                fp.write(buffer)

            records_metadata[rec_index] = SegyPartRecord(path=record_file_path, index=global_index)

    return records_metadata


def concat_files(paths: list[Path], progress: bool = False) -> Path:
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

    with first_file.open(mode="ab+") as first_fp:
        for next_file in paths:
            with next_file.open(mode="rb") as next_fp:
                copyfileobj(next_fp, first_fp)

            Path(next_file).unlink()

    return first_file
