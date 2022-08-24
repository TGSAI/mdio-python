"""Conversion from to MDIO various other formats."""


from __future__ import annotations

import uuid
from os import path

import numpy as np
from dask.array.core import Array
from dask.base import compute_as_if_collection
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from tqdm.dask import TqdmCallback

from mdio import MDIOReader
from mdio.segy._workers import write_block_to_segy
from mdio.segy.creation import mdio_spec_to_segy
from mdio.segy.creation import merge_partial_segy


try:
    import distributed
except ImportError:
    distributed = None


def mdio_to_segy(
    mdio_path_or_buffer: str,
    output_segy_path: str,
    endian: str = "big",
    access_pattern: str = "012",
    out_sample_format: str = "ibm32",
    storage_options: dict = None,
    new_chunks: tuple[int, ...] = None,
    selection_mask: np.ndarray = None,
    client: distributed.Client = None,
) -> None:
    """Convert MDIO file to SEG-Y format.

    MDIO allows exporting multidimensional seismic data back to the flattened
    seismic format SEG-Y, to be used in data transmission.

    The input headers are preserved as is, and will be transferred to the
    output file.

    The user has control over the endianness, and the floating point data
    type. However, by default we export as Big-Endian IBM float, per the
    SEG-Y format's default.

    The input MDIO can be local or cloud based. However, the output SEG-Y
    will be generated locally.

    A `selection_mask` can be provided (in the shape of the spatial grid)
    to export a subset of the seismic data.

    Args:
        mdio_path_or_buffer: Input path where the MDIO is located
        output_segy_path: Path to the output SEG-Y file
        endian: Endianness of the input SEG-Y. Rev.2 allows little
            endian. Default is 'big'.
        access_pattern: This specificies the chunk access pattern. Underlying
            zarr.Array must exist. Examples: '012', '01'
        out_sample_format: Output sample format.
            Currently support: {'ibm32', 'ieee32'}. Default is 'ibm32'.
        storage_options: Storage options for the cloud storage backend.
            Default: None (will assume anonymous access)
        new_chunks: Set manual chunksize. For development purposes only.
        selection_mask: Array that lists the subset of traces
        client: Dask client. If `None` we will use local threaded scheduler.
            If `auto` is used we will create multiple processes (with
            8 threads each).

    Raises:
        ImportError: if distributed package isn't installed but requested.
        ValueError: if cut mask is empty, i.e. no traces will be written.

    Examples:
        To export an existing local MDIO file to SEG-Y we use the code
        snippet below. This will export the full MDIO (without padding) to
        SEG-Y format using IBM floats and big-endian byte order.

        >>> from mdio import mdio_to_segy
        >>>
        >>>
        >>> mdio_to_segy(
        ...     mdio_path_or_buffer="prefix2/file.mdio",
        ...     output_segy_path="prefix/file.segy",
        ... )

        If we want to export this as an IEEE big-endian, using a selection
        mask, we would run:

        >>> mdio_to_segy(
        ...     mdio_path_or_buffer="prefix2/file.mdio",
        ...     output_segy_path="prefix/file.segy",
        ...     selection_mask=boolean_mask,
        ...     out_sample_format="ieee32",
        ... )

    """
    backend = "dask"

    mdio = MDIOReader(
        mdio_path_or_buffer=mdio_path_or_buffer,
        access_pattern=access_pattern,
        storage_options=storage_options,
    )

    ndim = mdio.n_dim

    # We flatten the z-axis (time or depth); so ieee2ibm, and byte-swaps etc
    # can run on big chunks of data.
    auto_chunk = (None,) * (ndim - 2) + ("100M",) + (-1,)
    new_chunks = new_chunks if new_chunks is not None else auto_chunk

    creation_args = [
        mdio_path_or_buffer,
        output_segy_path,
        endian,
        access_pattern,
        out_sample_format,
        storage_options,
        new_chunks,
        selection_mask,
        backend,
    ]

    if client is not None:
        if distributed is not None:
            # This is in case we work with big data
            feature = client.submit(mdio_spec_to_segy, *creation_args)
            mdio, sample_format = feature.result()
        else:
            msg = "Distributed client was provided, but `distributed` is not installed"
            raise ImportError(msg)
    else:
        mdio, sample_format = mdio_spec_to_segy(*creation_args)

    num_samp = mdio.shape[-1]

    live_mask = mdio.live_mask.compute()

    if selection_mask is not None:
        live_mask = live_mask & selection_mask

    # This handles the case if we are skipping a whole block.
    if live_mask.sum() == 0:
        raise ValueError("No traces will be written out. Live mask is empty.")

    # Find rough dim limits, so we don't unnecessarily hit disk / cloud store.
    # Typically, gets triggered when there is a selection mask
    dim_slices = ()
    live_nonzeros = live_mask.nonzero()
    for dim_nonzeros in live_nonzeros:
        start = np.min(dim_nonzeros)
        stop = np.max(dim_nonzeros) + 1
        dim_slices += (slice(start, stop),)

    # Lazily pull the data with limits now, and limit mask so its the same shape.
    live_mask, headers, traces = mdio[dim_slices]

    if selection_mask is not None:
        selection_mask = selection_mask[dim_slices]
        live_mask = live_mask & selection_mask

    # Now we flatten the data in the slowest changing axis (i.e. 0)
    # TODO: Add support for flipping these, if user wants
    axis = 0

    # Get new chunksizes for sequential array
    seq_trc_chunks = tuple(
        (dim_chunks if idx == axis else (sum(dim_chunks),))
        for idx, dim_chunks in enumerate(traces.chunks)
    )

    # We must unify chunks with "trc_chunks" here because
    # headers and live mask may have different chunking.
    # We don't take the time axis for headers / live
    # Still lazy computation
    traces_seq = traces.rechunk(seq_trc_chunks)
    headers_seq = headers.rechunk(seq_trc_chunks[:-1])
    live_seq = live_mask.rechunk(seq_trc_chunks[:-1])

    # Build a Dask graph to do the computation
    # Name of task. Using uuid1 is important because
    # we could potentially generate these from different machines
    task_name = "block-to-sgy-part-" + str(uuid.uuid1())

    trace_keys = flatten(traces_seq.__dask_keys__())
    header_keys = flatten(headers_seq.__dask_keys__())
    live_keys = flatten(live_seq.__dask_keys__())

    all_keys = zip(trace_keys, header_keys, live_keys)

    # tmp file root
    out_dir = path.dirname(output_segy_path)

    task_graph_dict = {}
    block_file_paths = []
    for idx, (trace_key, header_key, live_key) in enumerate(all_keys):
        block_file_name = f".{idx}_{uuid.uuid1()}._segyblock"
        block_file_path = path.join(out_dir, block_file_name)
        block_file_paths.append(block_file_path)

        block_args = (
            block_file_path,
            trace_key,
            header_key,
            live_key,
            num_samp,
            sample_format,
            endian,
        )

        task_graph_dict[(task_name, idx)] = (write_block_to_segy,) + block_args

    # Make actual graph
    task_graph = HighLevelGraph.from_collections(
        task_name,
        task_graph_dict,
        dependencies=[traces_seq, headers_seq, live_seq],
    )

    # Note this doesn't work with distributed.
    tqdm_kw = dict(unit="block", dynamic_ncols=True)
    block_progress = TqdmCallback(desc="Step 1 / 2 Writing Blocks", **tqdm_kw)

    with block_progress:
        block_exists = compute_as_if_collection(
            cls=Array,
            dsk=task_graph,
            keys=list(task_graph_dict),
            scheduler=client,
        )

    merge_args = [output_segy_path, block_file_paths, block_exists]
    if client is not None:
        _ = client.submit(merge_partial_segy, *merge_args).result()
    else:
        merge_partial_segy(*merge_args)
