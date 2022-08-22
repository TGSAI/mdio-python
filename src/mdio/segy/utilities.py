"""More utilities for reading SEG-Ys."""


from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt

from mdio.core import Dimension
from mdio.segy.parsers import parse_sample_axis
from mdio.segy.parsers import parse_trace_headers


def get_grid_plan(
    segy_path: str,
    segy_endian: str,
    index_bytes: Sequence[int],
    index_names: Sequence[str],
    index_lengths: Sequence[int],
    binary_header: dict,
    return_headers: bool = False,
) -> Union[List[Dimension], Tuple[List[Dimension], npt.ArrayLike]]:
    """Infer dimension ranges, and increments.

    Generates multiple dimensions with the following steps:
    1. Read index headers
    2. Get min, max, and increments
    3. Create `Dimension` with appropriate range, index, and description.
    4. Create `Dimension` for sample axis using binary header.

    Args:
        segy_path: Path to the input SEG-Y file
        segy_endian: Endianness of the input SEG-Y.
        index_bytes: Tuple of the byte location for the index attributes
        index_names: Tuple of the names for the index attributes
        index_lengths: Tuple of the byte lengths for the index attributes.
            Default will be 4-byte for all indices.
        binary_header: Dictionary containing binary header key, value pairs.
        return_headers: Option to return parsed headers with `Dimension` objects.
            Default is False.

    Returns:
        All index dimensions or dimensions together with header values.
    """
    index_dim = len(index_bytes)

    # Default is 4-byte for each index.
    index_lengths = [4] * index_dim if index_lengths is None else index_lengths

    index_headers = parse_trace_headers(
        segy_path=segy_path,
        segy_endian=segy_endian,
        byte_locs=index_bytes,
        byte_lengths=index_lengths,
    )

    if index_names is None:
        index_names = [f"index_{dim}" for dim in range(index_dim)]

    dims = []
    for dim, dim_name in enumerate(index_names):
        dim_unique = np.unique(index_headers[:, dim])
        dims.append(Dimension(coords=dim_unique, name=dim_name))

    sample_dim = parse_sample_axis(binary_header=binary_header)

    dims.append(sample_dim)

    return dims, index_headers if return_headers else dims
