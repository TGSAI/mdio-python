"""SEG-Y geometry handling functions."""

from __future__ import annotations

import logging
import time
from abc import ABC
from abc import abstractmethod
from enum import Enum
from enum import auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib import recfunctions as rfn

from mdio.segy.exceptions import GridOverrideKeysError
from mdio.segy.exceptions import GridOverrideMissingParameterError
from mdio.segy.exceptions import GridOverrideUnknownError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike
    from numpy.typing import NDArray
    from segy.arrays import HeaderArray

    from mdio.builder.templates.base import AbstractDatasetTemplate


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

        for gun in unique_guns_in_line:
            gun_mask = gun_current == gun
            shots_for_gun = shot_current[gun_mask]
            num_shots = np.unique(shots_for_gun).shape[0]
            mod_shots = np.floor(shots_for_gun / num_guns)
            if len(np.unique(mod_shots)) != num_shots:
                msg = "%s %s has %s shots; div by %s guns gives %s unique mod shots."
                logger.info(msg, line_field, line_val, num_shots, num_guns, len(np.unique(mod_shots)))
                geom_type = ShotGunGeometryType.A
                return unique_lines, unique_guns_per_line, geom_type

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


class GridOverrideCommand(ABC):
    """Abstract base class for grid override commands."""

    @property
    @abstractmethod
    def required_keys(self) -> set:
        """Get the set of required keys for the grid override command."""

    @property
    @abstractmethod
    def required_parameters(self) -> set:
        """Get the set of required parameters for the grid override command."""

    @abstractmethod
    def validate(
        self,
        index_headers: HeaderArray,
        grid_overrides: dict[str, bool | int],
        template: AbstractDatasetTemplate | None = None,
    ) -> None:
        """Validate if this transform should run on the type of data."""

    @abstractmethod
    def transform(
        self,
        index_headers: HeaderArray,
        grid_overrides: dict[str, bool | int],
        template: AbstractDatasetTemplate,  # noqa: ARG002
    ) -> NDArray:
        """Perform the grid transform."""

    def transform_index_names(self, index_names: Sequence[str]) -> Sequence[str]:
        """Perform the transform of index names.

        Optional method: Subclasses may override this method to provide custom behavior. If not
        overridden, this default implementation will be used, which is a no-op.

        Args:
            index_names: List of index names to be modified.

        Returns:
            New tuple of index names after the transform.
        """
        return index_names

    def transform_chunksize(
        self,
        chunksize: Sequence[int],
        grid_overrides: dict[str, bool | int],
    ) -> Sequence[int]:
        """Perform the transform of chunksize.

        Optional method: Subclasses may override this method to provide custom behavior. If not
        overridden, this default implementation will be used, which is a no-op.

        Args:
            chunksize: List of chunk sizes to be modified.
            grid_overrides: Full grid override parameterization.

        Returns:
            New tuple of chunk sizes after the transform.
        """
        _ = grid_overrides  # Unused, required for ABC compatibility
        return chunksize

    @property
    def name(self) -> str:
        """Convenience property to get the name of the command."""
        return self.__class__.__name__

    def check_required_keys(self, index_headers: HeaderArray) -> None:
        """Check if all required keys are present in the index headers."""
        index_names = index_headers.dtype.names
        if not self.required_keys.issubset(index_names):
            raise GridOverrideKeysError(self.name, self.required_keys)

    def check_required_params(self, grid_overrides: dict[str, str | int]) -> None:
        """Check if all required keys are present in the index headers."""
        if self.required_parameters is None:
            return

        passed_parameters = set(grid_overrides.keys())

        if not self.required_parameters.issubset(passed_parameters):
            missing_params = self.required_parameters - passed_parameters
            raise GridOverrideMissingParameterError(self.name, missing_params)


class DuplicateIndex(GridOverrideCommand):
    """Automatically handle duplicate traces in a new axis - trace with chunksize 1."""

    required_keys = None
    required_parameters = None

    def validate(
        self,
        index_headers: HeaderArray,
        grid_overrides: dict[str, bool | int],
        template: AbstractDatasetTemplate | None = None,  # noqa: ARG002
    ) -> None:
        """Validate if this transform should run on the type of data."""
        if self.required_keys is not None:
            self.check_required_keys(index_headers)
        self.check_required_params(grid_overrides)

    def transform(
        self,
        index_headers: HeaderArray,
        grid_overrides: dict[str, bool | int],
        template: AbstractDatasetTemplate,
    ) -> NDArray:
        """Perform the grid transform."""
        self.validate(index_headers, grid_overrides)

        # Filter out coordinate fields, keep only dimensions for trace indexing
        coord_fields = set(template.coordinate_names) if template else set()

        # For NonBinned: non_binned_dims should be excluded from trace indexing grouping
        # because they become coordinates indexed by the trace dimension, not grouping keys.
        # The trace index should count all traces per remaining dimension combination.
        non_binned_dims = set(grid_overrides.get("non_binned_dims", [])) if grid_overrides else set()

        dim_fields = [
            name
            for name in index_headers.dtype.names
            if name != "trace" and name not in coord_fields and name not in non_binned_dims
        ]

        # Create trace indices on dimension fields only
        dim_headers = index_headers[dim_fields] if dim_fields else index_headers
        dim_headers_with_trace = analyze_non_indexed_headers(dim_headers)

        # Add trace field back to full headers
        if dim_headers_with_trace is not None and "trace" in dim_headers_with_trace.dtype.names:
            trace_values = np.array(dim_headers_with_trace["trace"])
            index_headers = rfn.append_fields(index_headers, "trace", trace_values, usemask=False)

        return index_headers

    def transform_index_names(self, index_names: Sequence[str]) -> Sequence[str]:
        """Insert dimension "trace" to the sample-1 dimension."""
        new_names = list(index_names)
        new_names.append("trace")
        return tuple(new_names)

    def transform_chunksize(
        self,
        chunksize: Sequence[int],
        grid_overrides: dict[str, bool | int],
    ) -> Sequence[int]:
        """Insert chunksize of 1 to the sample-1 dimension."""
        _ = grid_overrides  # Unused, required for ABC compatibility
        new_chunks = list(chunksize)
        new_chunks.insert(-1, 1)
        return tuple(new_chunks)


class NonBinned(DuplicateIndex):
    """Handle non-binned dimensions by converting them to a trace dimension with coordinates.

    This override takes dimensions that are not regularly sampled (non-binned) and converts
    them into a single 'trace' dimension. The original non-binned dimensions become coordinates
    indexed by the trace dimension.

    Example:
        Template with dimensions [shot_point, cable, channel, azimuth, offset, sample]
        and non_binned_dims=['azimuth', 'offset'] becomes:
        - dimensions: [shot_point, cable, channel, trace, sample]
        - coordinates: azimuth and offset with dimensions [shot_point, cable, channel, trace]

    Attributes:
        required_keys: No required keys for this override.
        required_parameters: Set containing 'chunksize' and 'non_binned_dims'.
    """

    required_keys = None
    required_parameters = {"chunksize", "non_binned_dims"}

    def validate(
        self,
        index_headers: HeaderArray,
        grid_overrides: dict[str, bool | int],
        template: AbstractDatasetTemplate | None = None,  # noqa: ARG002
    ) -> None:
        """Validate if this transform should run on the type of data."""
        self.check_required_params(grid_overrides)

        # Validate that non_binned_dims is a list
        non_binned_dims = grid_overrides.get("non_binned_dims", [])
        if not isinstance(non_binned_dims, list):
            msg = f"non_binned_dims must be a list, got {type(non_binned_dims)}"
            raise ValueError(msg)

        # Validate that all non-binned dimensions exist in headers
        missing_dims = set(non_binned_dims) - set(index_headers.dtype.names)
        if missing_dims:
            msg = f"Non-binned dimensions {missing_dims} not found in index headers"
            raise ValueError(msg)

    def transform_chunksize(
        self,
        chunksize: Sequence[int],
        grid_overrides: dict[str, bool | int],
    ) -> Sequence[int]:
        """Insert chunksize for trace dimension at N-1 position."""
        new_chunks = list(chunksize)
        trace_chunksize = grid_overrides["chunksize"]
        new_chunks.insert(-1, trace_chunksize)
        return tuple(new_chunks)


class AutoChannelWrap(GridOverrideCommand):
    """Automatically determine Streamer acquisition type."""

    required_keys = {"shot_point", "cable", "channel"}
    required_parameters = None

    def validate(
        self,
        index_headers: HeaderArray,
        grid_overrides: dict[str, bool | int],
        template: AbstractDatasetTemplate | None = None,  # noqa: ARG002
    ) -> None:
        """Validate if this transform should run on the type of data."""
        self.check_required_keys(index_headers)
        self.check_required_params(grid_overrides)

    def transform(
        self,
        index_headers: HeaderArray,
        grid_overrides: dict[str, bool | int],
        template: AbstractDatasetTemplate,  # noqa: ARG002
    ) -> NDArray:
        """Perform the grid transform."""
        self.validate(index_headers, grid_overrides)

        result = analyze_streamer_headers(index_headers)
        unique_cables, cable_chan_min, cable_chan_max, geom_type = result
        logger.info("Ingesting dataset as %s", geom_type.name)

        for cable, chan_min, chan_max in zip(unique_cables, cable_chan_min, cable_chan_max, strict=True):
            logger.info("Cable: %s has min chan: %s and max chan: %s", cable, chan_min, chan_max)

        # This might be slow and could be improved with a rewrite to prevent so many lookups
        if geom_type == StreamerShotGeometryType.B:
            for idx, cable in enumerate(unique_cables):
                cable_idxs = np.where(index_headers["cable"][:] == cable)
                cc_min = cable_chan_min[idx]

                index_headers["channel"][cable_idxs] = index_headers["channel"][cable_idxs] - cc_min + 1

        return index_headers


class CalculateShotIndex(GridOverrideCommand):
    """Calculate dense shot_index from shot_point values for OBN templates.

    Creates a 0-based shot_index dimension from sparse or interleaved shot_point
    values, grouping by shot_line and gun. This is required for the OBN template
    which uses shot_index as a calculated dimension.

    Required headers: shot_line, gun, shot_point

    Attributes:
        required_parameters: Set of required parameters (None for this class).
    """

    required_parameters = None

    @property
    def required_keys(self) -> set:
        """Return required header keys for OBN shot index calculation."""
        return {"shot_line", "gun", "shot_point"}

    def validate(
        self,
        index_headers: HeaderArray,
        grid_overrides: dict[str, bool | int],
        template: AbstractDatasetTemplate | None = None,
    ) -> None:
        """Validate if this transform should run on the type of data."""
        # Import here to avoid circular imports at module load time
        from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate  # noqa: PLC0415

        if template is None:
            msg = "CalculateShotIndex requires a template."
            raise TypeError(msg)

        if not isinstance(template, Seismic3DObnReceiverGathersTemplate):
            msg = (
                f"CalculateShotIndex only supports Seismic3DObnReceiverGathersTemplate, got {type(template).__name__}."
            )
            raise TypeError(msg)

        index_names = set(index_headers.dtype.names)
        if not self.required_keys.issubset(index_names):
            raise GridOverrideKeysError(self.name, self.required_keys)
        self.check_required_params(grid_overrides)

    def transform(
        self,
        index_headers: HeaderArray,
        grid_overrides: dict[str, bool | int],
        template: AbstractDatasetTemplate,
    ) -> NDArray:
        """Perform the grid transform to calculate shot_index from shot_point."""
        self.validate(index_headers, grid_overrides, template)

        line_field = "shot_line"
        result = analyze_lines_for_guns(index_headers, line_field=line_field)
        unique_lines, unique_guns_per_line, geom_type = result
        logger.info("Ingesting OBN dataset as shot type: %s", geom_type.name)

        max_num_guns = 1
        for line_val in unique_lines:
            guns = unique_guns_per_line[str(line_val)]
            logger.info("%s: %s has guns: %s", line_field, line_val, guns)
            max_num_guns = max(len(guns), max_num_guns)

        # Only calculate shot_index when shot points are interleaved across guns (Type B)
        if geom_type == ShotGunGeometryType.B:
            shot_index = np.empty(len(index_headers), dtype="uint32")
            # Use .base if available (view of another array), otherwise use the array directly
            base_array = index_headers.base if index_headers.base is not None else index_headers
            index_headers = rfn.append_fields(base_array, "shot_index", shot_index)

            for line_val in unique_lines:
                line_idxs = np.where(index_headers[line_field][:] == line_val)
                index_headers["shot_index"][line_idxs] = np.floor(index_headers["shot_point"][line_idxs] / max_num_guns)
                # Make shot index zero-based PER line
                index_headers["shot_index"][line_idxs] -= index_headers["shot_index"][line_idxs].min()

        return index_headers


class AutoShotWrap(GridOverrideCommand):
    """Automatic shot index calculation from interleaved shot points for Streamer templates.

    This grid override handles multi-gun acquisition where shot points may be
    interleaved across guns. It calculates a dense shot_index from sparse shot_point values.

    Supported Templates:
        - Seismic3DStreamerFieldRecordsTemplate: Uses sail_line, requires cable/channel

    Note:
        For OBN templates, use CalculateShotIndex instead.

    Attributes:
        required_parameters: Set of required parameters (None for this class).
    """

    required_parameters = None

    @property
    def required_keys(self) -> set:
        """Return required header keys for streamer shot index calculation."""
        return {"sail_line", "gun", "shot_point", "cable", "channel"}

    def validate(
        self,
        index_headers: HeaderArray,
        grid_overrides: dict[str, bool | int],
        template: AbstractDatasetTemplate | None = None,
    ) -> None:
        """Validate if this transform should run on the type of data."""
        # Import here to avoid circular imports at module load time
        from mdio.builder.templates.seismic_3d_streamer_field import (  # noqa: PLC0415
            Seismic3DStreamerFieldRecordsTemplate,
        )

        if template is None:
            msg = "AutoShotWrap requires a template."
            raise TypeError(msg)

        if not isinstance(template, Seismic3DStreamerFieldRecordsTemplate):
            msg = (
                f"AutoShotWrap only supports Seismic3DStreamerFieldRecordsTemplate, "
                f"got {type(template).__name__}. For OBN templates, use CalculateShotIndex."
            )
            raise TypeError(msg)

        index_names = set(index_headers.dtype.names)
        if not self.required_keys.issubset(index_names):
            raise GridOverrideKeysError(self.name, self.required_keys)
        self.check_required_params(grid_overrides)

    def transform(
        self,
        index_headers: HeaderArray,
        grid_overrides: dict[str, bool | int],
        template: AbstractDatasetTemplate,
    ) -> NDArray:
        """Perform the grid transform to calculate shot_index from shot_point."""
        self.validate(index_headers, grid_overrides, template)

        line_field = "sail_line"
        result = analyze_lines_for_guns(index_headers, line_field=line_field)
        unique_lines, unique_guns_per_line, geom_type = result
        logger.info("Ingesting streamer dataset as shot type: %s", geom_type.name)

        max_num_guns = 1
        for line_val in unique_lines:
            guns = unique_guns_per_line[str(line_val)]
            logger.info("%s: %s has guns: %s", line_field, line_val, guns)
            max_num_guns = max(len(guns), max_num_guns)

        # Only calculate shot_index when shot points are interleaved across guns (Type B)
        if geom_type == ShotGunGeometryType.B:
            shot_index = np.empty(len(index_headers), dtype="uint32")
            # Use .base if available (view of another array), otherwise use the array directly
            base_array = index_headers.base if index_headers.base is not None else index_headers
            index_headers = rfn.append_fields(base_array, "shot_index", shot_index)

            for line_val in unique_lines:
                line_idxs = np.where(index_headers[line_field][:] == line_val)
                index_headers["shot_index"][line_idxs] = np.floor(index_headers["shot_point"][line_idxs] / max_num_guns)
                # Make shot index zero-based PER line
                index_headers["shot_index"][line_idxs] -= index_headers["shot_index"][line_idxs].min()

        return index_headers


class GridOverrider:
    """Executor for grid overrides.

    We support a certain type of grid overrides, and they have to be implemented following the
    ABC's in this module.

    This class applies the grid overrides if needed.
    """

    def __init__(self) -> None:
        self.commands = {
            "AutoChannelWrap": AutoChannelWrap(),
            "AutoShotWrap": AutoShotWrap(),
            "CalculateShotIndex": CalculateShotIndex(),
            "NonBinned": NonBinned(),
            "HasDuplicates": DuplicateIndex(),
        }

        self.parameters = self.get_allowed_parameters()

    def get_allowed_parameters(self) -> set:
        """Get list of allowed parameters from the allowed commands."""
        parameters = set()
        for command in self.commands.values():
            if command.required_parameters is None:
                continue

            parameters.update(command.required_parameters)

        # Add optional parameters that are not strictly required but are valid
        parameters.add("non_binned_dims")

        return parameters

    def _synthesize_obn_component(
        self,
        index_headers: HeaderArray,
        template: AbstractDatasetTemplate | None,
    ) -> HeaderArray:
        """Synthesize component field for OBN template when missing from headers.

        OBN data may not have a component field in the SEG-Y headers (e.g., single-component
        data). When using Seismic3DObnReceiverGathersTemplate and component is missing,
        this method synthesizes it with a constant value of 1.

        Args:
            index_headers: The parsed index headers from SEG-Y.
            template: The dataset template.

        Returns:
            Headers with component field added if applicable.
        """
        # Import here to avoid circular imports at module load time
        from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate  # noqa: PLC0415

        if not isinstance(template, Seismic3DObnReceiverGathersTemplate):
            return index_headers

        if "component" in index_headers.dtype.names:
            return index_headers

        logger.warning(
            "SEG-Y headers do not contain 'component' field required by template '%s'. "
            "Synthesizing 'component' dimension with constant value 1 for all traces.",
            template.name,
        )
        synthetic_col = np.full(len(index_headers), 1, dtype=np.int32)
        base_array = index_headers.base if index_headers.base is not None else index_headers
        return rfn.append_fields(base_array, "component", synthetic_col, usemask=False)

    def run(
        self,
        index_headers: HeaderArray,
        index_names: Sequence[str],
        grid_overrides: dict[str, bool],
        chunksize: Sequence[int] | None = None,
        template: AbstractDatasetTemplate | None = None,
    ) -> tuple[HeaderArray, tuple[str], tuple[int]]:
        """Run grid overrides and return result."""
        # Synthesize component for OBN template if missing
        index_headers = self._synthesize_obn_component(index_headers, template)

        for override in grid_overrides:
            if override in self.parameters:
                continue

            if override not in self.commands:
                raise GridOverrideUnknownError(override)

            function = self.commands[override].transform
            index_headers = function(index_headers, grid_overrides=grid_overrides, template=template)

            function = self.commands[override].transform_index_names
            index_names = function(index_names)

            function = self.commands[override].transform_chunksize
            chunksize = function(chunksize, grid_overrides=grid_overrides)

        return index_headers, index_names, chunksize
