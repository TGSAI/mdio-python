"""Conversion from to MDIO various other formats."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from psutil import cpu_count
from tqdm.dask import TqdmCallback

from mdio.api.opener import open_dataset
from mdio.segy.blocked_io import to_segy
from mdio.segy.creation import concat_files
from mdio.segy.creation import mdio_spec_to_segy
from mdio.segy.utilities import segy_export_rechunker

try:
    import distributed
except ImportError:
    distributed = None

if TYPE_CHECKING:
    from segy.schema import SegySpec

    from mdio.core.storage_location import StorageLocation

default_cpus = cpu_count(logical=True)
NUM_CPUS = int(os.getenv("MDIO__EXPORT__CPU_COUNT", default_cpus))


def mdio_to_segy(  # noqa: PLR0912, PLR0913, PLR0915
    segy_spec: SegySpec,
    input_location: StorageLocation,
    output_location: StorageLocation,
    selection_mask: np.ndarray = None,
    client: distributed.Client = None,
) -> None:
    """Convert MDIO file to SEG-Y format.

    We export N-D seismic data to the flattened SEG-Y format used in data transmission.

    The input headers are preserved as is, and will be transferred to the output file.

    Input MDIO can be local or cloud based. However, the output SEG-Y will be generated locally.

    A `selection_mask` can be provided (same shape as spatial grid) to export a subset.

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        input_location: Store or URL (and cloud options) for MDIO file.
        output_location: Path to the output SEG-Y file.
        selection_mask: Array that lists the subset of traces
        client: Dask client. If `None` we will use local threaded scheduler. If `auto` is used we
            will create multiple processes (with 8 threads each).

    Raises:
        ImportError: if distributed package isn't installed but requested.
        ValueError: if cut mask is empty, i.e. no traces will be written.

    Examples:
        To export an existing local MDIO file to SEG-Y we use the code snippet below. This will
        export the full MDIO (without padding) to SEG-Y format.

        >>> from mdio import mdio_to_segy, StorageLocation
        >>>
        >>> input_location = StorageLocation("prefix2/file.mdio")
        >>> output_location = StorageLocation("prefix/file.segy")
        >>> mdio_to_segy(input_location, output_location)
    """
    output_segy_path = Path(output_location.uri)

    # First we open with vanilla zarr backend and then get some info
    # We will re-open with `new_chunks` and Dask later in mdio_spec_to_segy
    dataset = open_dataset(input_location)

    trace_variable_name = dataset.attrs["attributes"]["traceVariableName"]
    amplitude = dataset[trace_variable_name]
    chunks = amplitude.encoding["preferred_chunks"]
    sizes = amplitude.sizes
    dtype = amplitude.dtype
    new_chunks = segy_export_rechunker(chunks, sizes, dtype)

    creation_args = [segy_spec, input_location, output_location, new_chunks]

    if client is not None:
        if distributed is not None:
            # This is in case we work with big data
            feature = client.submit(mdio_spec_to_segy, *creation_args)
            dataset, segy_factory = feature.result()
        else:
            msg = "Distributed client was provided, but `distributed` is not installed"
            raise ImportError(msg)
    else:
        dataset, segy_factory = mdio_spec_to_segy(*creation_args)

    trace_mask = dataset["trace_mask"].compute()

    if selection_mask is not None:
        if trace_mask.shape != selection_mask.shape:
            msg = "Selection mask and trace mask shapes do not match."
            raise ValueError(msg)
        selection_mask = trace_mask.copy(data=selection_mask)  # make into DataArray
        trace_mask = trace_mask & selection_mask

    # This handles the case if we are skipping a whole block.
    if trace_mask.sum() == 0:
        msg = "No traces will be written out. Live mask is empty."
        raise ValueError(msg)

    # Find rough dim limits, so we don't unnecessarily hit disk / cloud store.
    # Typically, gets triggered when there is a selection mask
    dim_slices = {}
    dim_live_indices = np.nonzero(trace_mask.values)
    for dim_name, dim_live in zip(trace_mask.dims, dim_live_indices, strict=True):
        start = dim_live.min().item()
        stop = dim_live.max().item() + 1
        dim_slices[dim_name] = slice(start, stop)

    # Lazily pull the data with limits now.
    # All the variables, metadata, etc. is all sliced to the same range.
    dataset = dataset.isel(dim_slices)

    if selection_mask is not None:
        selection_mask = selection_mask[dim_slices]
        dataset["trace_mask"] = dataset["trace_mask"] & selection_mask

    # tmp file root
    out_dir = output_segy_path.parent
    tmp_dir = TemporaryDirectory(dir=out_dir)

    with tmp_dir:
        with TqdmCallback(desc="Unwrapping MDIO Blocks"):
            block_records = to_segy(
                samples=dataset[trace_variable_name].data,
                headers=dataset["headers"].data,
                live_mask=dataset["trace_mask"].data,
                segy_factory=segy_factory,
                file_root=tmp_dir.name,
            )

            if client is not None:
                block_records = block_records.compute()
            else:
                block_records = block_records.compute(num_workers=NUM_CPUS)

        ordered_files = [rec.path for rec in block_records.ravel() if rec != 0]
        ordered_files = [output_segy_path] + ordered_files

        if client is not None:
            _ = client.submit(concat_files, paths=ordered_files).result()
        else:
            concat_files(paths=ordered_files, progress=True)
