"""SEG-Y file information utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mdio.segy.scalar import _get_coordinate_scalar
from mdio.segy.segy_file_async import SegyFileArguments
from mdio.segy.segy_file_async import SegyFileAsync

__all__ = [
    "SegyFileInfo",
    "get_segy_file_info",
]


@dataclass
class SegyFileInfo:
    """SEG-Y file header information."""

    num_traces: int
    sample_labels: NDArray[np.int32]
    text_header: str
    binary_header_dict: dict
    raw_binary_headers: bytes
    coordinate_scalar: int


def get_segy_file_info(segy_file_kwargs: SegyFileArguments) -> SegyFileInfo:
    """Reads information from a SEG-Y file.

    Args:
        segy_file_kwargs: Arguments to open SegyFile instance.

    Returns:
        SegyFileInfo containing number of traces, sample labels, and header info.
    """
    segy_file = SegyFileAsync(**segy_file_kwargs)
    num_traces = segy_file.num_traces
    sample_labels = segy_file.sample_labels

    text_header = segy_file.text_header

    # Get header information directly
    raw_binary_headers = segy_file.fs.read_block(
        fn=segy_file.url,
        offset=segy_file.spec.binary_header.offset,
        length=segy_file.spec.binary_header.itemsize,
    )

    # We read here twice, but it's ok for now. Only 400-bytes.
    binary_header_dict = segy_file.binary_header.to_dict()

    coordinate_scalar = _get_coordinate_scalar(segy_file)

    return SegyFileInfo(
        num_traces=num_traces,
        sample_labels=sample_labels,
        text_header=text_header,
        binary_header_dict=binary_header_dict,
        raw_binary_headers=raw_binary_headers,
        coordinate_scalar=coordinate_scalar,
    )
