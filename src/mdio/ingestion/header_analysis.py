"""Header analysis utilities for MDIO ingestion.

This module contains functions for analyzing SEG-Y headers to determine
geometry and create indices for various acquisition types.
"""

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
    """Enumerates streamer shot geometry types by channel numbering.

    Type A: Channels restart numbering for each cable (1 to N per cable).
    Type B: Channels are numbered sequentially across all cables.
    Type C: Channels are numbered in reverse sequential order (Type B reversed).
    """

    A = auto()
    B = auto()
    C = auto()


class ShotGunGeometryType(Enum):
    """Enumerates acquisition gun geometries for shot data.

    Single: One source (gun) per shot.
    Alternate: Alternating sources (e.g., dual gun flipping).
    Simultaneous: Multiple sources fire at the same time, identified by unique (shot point, gun) pairs.
    """

    SINGLE = auto()
    ALTERNATE = auto()
    SIMULTANEOUS = auto()


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
    # Find unique cable ids
    unique_cables = np.sort(np.unique(index_headers["cable"]))

    # Find channel min and max values for each cable
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


def analyze_saillines_for_guns(
    index_headers: HeaderArray,
) -> tuple[NDArray, dict[str, list], ShotGunGeometryType]:
    """Check input headers for SEG-Y input to help determine geometry of shots and guns.

    This function reads in trace_qc_count headers and finds the unique gun values. The function
    then checks to ensure shot numbers are dense.

    Args:
        index_headers: numpy array with index headers

    Returns:
        tuple of unique_sail_lines, unique_guns_in_sail_line, geom_type
    """
    # Find unique cable ids
    unique_sail_lines = np.sort(np.unique(index_headers["sail_line"]))
    unique_guns = np.sort(np.unique(index_headers["gun"]))
    logger.info("unique_sail_lines: %s", unique_sail_lines)
    logger.info("unique_guns: %s", unique_guns)

    # Find channel min and max values for each cable
    unique_guns_in_sail_line = {}

    geom_type = ShotGunGeometryType.B
    # Check shot numbers are still unique if div/num_guns
    for sail_line in unique_sail_lines:
        sail_line_mask = index_headers["sail_line"] == sail_line
        shot_current_sl = index_headers["shot_point"][sail_line_mask]
        gun_current_sl = index_headers["gun"][sail_line_mask]

        unique_guns_sl = np.sort(np.unique(gun_current_sl))
        num_guns_sl = unique_guns_sl.shape[0]
        unique_guns_in_sail_line[str(sail_line)] = list(unique_guns_sl)

        for gun in unique_guns_sl:
            gun_mask = gun_current_sl == gun
            shots_current_sl_gun = shot_current_sl[gun_mask]
            num_shots_sl = np.unique(shots_current_sl_gun).shape[0]
            mod_shots = np.floor(shots_current_sl_gun / num_guns_sl)
            if len(np.unique(mod_shots)) != num_shots_sl:
                msg = "Shot line %s has %s when using div by %s %s has %s unique mod shots."
                logger.info(msg, sail_line, num_shots_sl, num_guns_sl, np.unique(mod_shots))
                geom_type = ShotGunGeometryType.A
                return unique_sail_lines, unique_guns_in_sail_line, geom_type
    return unique_sail_lines, unique_guns_in_sail_line, geom_type


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
            unique_headers[header_key] = np.sort(np.unique(index_headers[header_key]))
            header_names.append(header_key)
            total_depth += 1

    counter = create_counter(0, total_depth, unique_headers, header_names)

    index_headers = create_trace_index(total_depth, counter, index_headers, header_names, dtype=dtype)

    t_stop = time.perf_counter()
    logger.debug("Time spent generating trace index: %.4f s", t_start - t_stop)
    return index_headers
