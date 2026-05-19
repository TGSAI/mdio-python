"""SEG-Y header analysis primitives for acquisition-geometry detection."""

from __future__ import annotations

import logging
import time
from enum import Enum
from enum import auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib import recfunctions as rfn

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from numpy.typing import NDArray
    from segy.arrays import HeaderArray


logger = logging.getLogger(__name__)


class StreamerShotGeometryType(Enum):
    r"""Shot geometry template types for streamer acquisition.

    Configuration A:
        Cable 1 ->          1------------------20
        Cable 2 ->         1-----------------20
        .                 1-----------------20
        .          ⛴ ☆  1-----------------20
        .                 1-----------------20
        Cable 6 ->         1-----------------20
        Cable 7 ->          1-----------------20


    Configuration B:
        Cable 1 ->          1------------------20
        Cable 2 ->         21-----------------40
        .                 41-----------------60
        .          ⛴ ☆  61-----------------80
        .                 81----------------100
        Cable 6 ->         101---------------120
        Cable 7 ->          121---------------140

    Configuration C:
        Cable ? ->        / 1------------------20
        Cable ? ->       / 21-----------------40
        .               / 41-----------------60
        .          ⛴ ☆ - 61-----------------80
        .               \ 81----------------100
        Cable ? ->       \ 101---------------120
        Cable ? ->        \ 121---------------140
    """

    A = auto()
    B = auto()
    C = auto()


class ShotGunGeometryType(Enum):
    r"""Shot geometry template types for multi-gun acquisition.

    For shot lines with multiple guns, we can have two configurations for numbering shot_point. The
    desired index is to have the shot point index for a given gun to be dense and unique
    (configuration A). Typically the shot_point is unique for the line and therefore is not dense
    for each gun (configuration B).

    Configuration A:
        Gun 1 ->         1------------------20
        Gun 2 ->         1------------------20

    Configuration B:
        Gun 1 ->         1------------------39
        Gun 2 ->         2------------------40

    """

    A = auto()
    B = auto()


def analyze_streamer_headers(
    index_headers: HeaderArray,
) -> tuple[NDArray, NDArray, NDArray, StreamerShotGeometryType]:
    """Check input headers for SEG-Y input to help determine geometry.

    This function reads in trace_qc_count headers and finds the unique cable values. The function
    then checks to ensure channel numbers for different cables do not overlap.

    Args:
        index_headers: numpy array with index headers

    Returns:
        tuple of unique_cables, cable_chan_min, cable_chan_max, geom_type
    """
    unique_cables = np.sort(np.unique(index_headers["cable"]))

    cable_chan_min = np.empty(unique_cables.shape)
    cable_chan_max = np.empty(unique_cables.shape)

    for idx, cable in enumerate(unique_cables):
        cable_mask = index_headers["cable"] == cable
        current_cable = index_headers["channel"][cable_mask]

        cable_chan_min[idx] = np.min(current_cable)
        cable_chan_max[idx] = np.max(current_cable)

    # Check channel numbers do not overlap for case B
    geom_type = StreamerShotGeometryType.B

    for idx1, cable1 in enumerate(unique_cables):
        min_val1 = cable_chan_min[idx1]
        max_val1 = cable_chan_max[idx1]

        cable1_range = (min_val1, max_val1)
        for idx2, cable2 in enumerate(unique_cables):
            if cable2 == cable1:
                continue

            min_val2 = cable_chan_min[idx2]
            max_val2 = cable_chan_max[idx2]
            cable2_range = (min_val2, max_val2)

            # Check for overlap and return early with Type A
            if min_val2 < max_val1 and max_val2 > min_val1:
                geom_type = StreamerShotGeometryType.A

                logger.info("Found overlapping channels, assuming streamer type A")
                overlap_info = (
                    "Cable %s index %s with channel range %s overlaps cable %s index %s with "
                    "channel range %s. Check for aux trace issues if the overlap is unexpected. "
                    "To fix, modify the SEG-Y file or use AutoIndex grid override (not channel) "
                    "for channel number correction."
                )
                logger.info(overlap_info, cable1, idx1, cable1_range, cable2, idx2, cable2_range)

                return unique_cables, cable_chan_min, cable_chan_max, geom_type

    return unique_cables, cable_chan_min, cable_chan_max, geom_type


def analyze_lines_for_guns(
    index_headers: HeaderArray,
    line_field: str = "sail_line",
) -> tuple[NDArray, dict[str, list], ShotGunGeometryType]:
    """Check input headers for SEG-Y input to help determine geometry of shots and guns.

    This is a generalized function that works with any line field name (sail_line, shot_line, etc.)
    to analyze multi-gun acquisition geometry and determine if shot points are interleaved.

    Args:
        index_headers: Numpy array with index headers.
        line_field: Name of the line field to use (e.g., 'sail_line', 'shot_line').

    Returns:
        tuple of (unique_lines, unique_guns_per_line, geom_type)
    """
    unique_lines = np.sort(np.unique(index_headers[line_field]))
    unique_guns = np.sort(np.unique(index_headers["gun"]))
    logger.info("unique_%s values: %s", line_field, unique_lines)
    logger.info("unique_guns: %s", unique_guns)

    unique_guns_per_line = {}

    geom_type = ShotGunGeometryType.B
    # Check shot numbers are still unique if div/num_guns
    for line_val in unique_lines:
        line_mask = index_headers[line_field] == line_val
        shot_current = index_headers["shot_point"][line_mask]
        gun_current = index_headers["gun"][line_mask]

        unique_guns_in_line = np.sort(np.unique(gun_current))
        num_guns = unique_guns_in_line.shape[0]
        unique_guns_per_line[str(line_val)] = list(unique_guns_in_line)

        # Skip gemoetry detection if we arlready know it's Type A
        if geom_type == ShotGunGeometryType.A:
            continue

        for gun in unique_guns_in_line:
            gun_mask = gun_current == gun
            shots_for_gun = shot_current[gun_mask]
            num_shots = np.unique(shots_for_gun).shape[0]
            mod_shots = np.floor(shots_for_gun / num_guns)
            if len(np.unique(mod_shots)) != num_shots:
                msg = "%s %s has %s shots; div by %s guns gives %s unique mod shots."
                logger.info(msg, line_field, line_val, num_shots, num_guns, len(np.unique(mod_shots)))
                geom_type = ShotGunGeometryType.A
                break # No need to check more guns for this line

    return unique_lines, unique_guns_per_line, geom_type


# Backward-compatible aliases for existing code
def analyze_saillines_for_guns(
    index_headers: HeaderArray,
) -> tuple[NDArray, dict[str, list], ShotGunGeometryType]:
    """Analyze sail lines for gun geometry. See analyze_lines_for_guns for details."""
    return analyze_lines_for_guns(index_headers, line_field="sail_line")


def create_counter(
    depth: int,
    total_depth: int,
    unique_headers: dict[str, NDArray],
    header_names: list[str],
) -> dict[str, dict]:
    """Helper function to create dictionary tree for counting trace key for auto index."""
    if depth == total_depth:
        return 0

    counter = {}

    header_key = header_names[depth]
    for header in unique_headers[header_key]:
        counter[header] = create_counter(depth + 1, total_depth, unique_headers, header_names)

    return counter


def create_trace_index(
    depth: int,
    counter: dict,
    index_headers: HeaderArray,
    header_names: list[str],
    dtype: DTypeLike = np.int16,
) -> NDArray | None:
    """Update dictionary counter tree for counting trace key for auto index."""
    if depth == 0:
        # If there's no hierarchical depth, no tracing needed.
        return None

    # Add index header
    trace_no_field = np.zeros(index_headers.shape, dtype=dtype)
    index_headers = rfn.append_fields(index_headers, "trace", trace_no_field, usemask=False)

    # Extract the relevant columns upfront
    headers = [index_headers[name] for name in header_names[:depth]]
    for idx, idx_values in enumerate(zip(*headers, strict=True)):
        if depth == 1:
            counter[idx_values[0]] += 1
            index_headers["trace"][idx] = counter[idx_values[0]]
        else:
            sub_counter = counter
            for idx_value in idx_values[:-1]:
                sub_counter = sub_counter[idx_value]
            sub_counter[idx_values[-1]] += 1
            index_headers["trace"][idx] = sub_counter[idx_values[-1]]

    return index_headers


def analyze_non_indexed_headers(index_headers: HeaderArray, dtype: DTypeLike = np.int16) -> NDArray:
    """Check input headers for SEG-Y input to help determine geometry.

    This function reads in trace_qc_count headers and finds the unique cable values. Then, it
    checks to make sure channel numbers for different cables do not overlap.

    Args:
        index_headers: numpy array with index headers
        dtype: numpy type for value of created trace header.

    Returns:
        Dict container header name as key and numpy array of values as value
    """
    # Find unique cable ids
    t_start = time.perf_counter()
    unique_headers = {}
    total_depth = 0
    header_names = []
    for header_key in index_headers.dtype.names:
        if header_key != "trace":
            unique_vals = np.sort(np.unique(index_headers[header_key]))
            unique_headers[header_key] = unique_vals
            header_names.append(header_key)
            total_depth += 1

    counter = create_counter(0, total_depth, unique_headers, header_names)

    index_headers = create_trace_index(total_depth, counter, index_headers, header_names, dtype=dtype)

    t_stop = time.perf_counter()
    logger.debug("Time spent generating trace index: %.4f s", t_start - t_stop)
    return index_headers
