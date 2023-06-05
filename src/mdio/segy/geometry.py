"""SEG-Y geometry handling functions."""


from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from enum import Enum
from enum import auto

import numpy as np
import numpy.typing as npt

from mdio.segy.exceptions import GridOverrideIncompatibleError
from mdio.segy.exceptions import GridOverrideKeysError
from mdio.segy.exceptions import GridOverrideMissingParameterError
from mdio.segy.exceptions import GridOverrideUnknownError


logger = logging.getLogger(__name__)


class StreamerShotGeometryType(Enum):
    r"""Shot  geometry template types for streamer acquisition.

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


def analyze_streamer_headers(
    index_headers: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, StreamerShotGeometryType]:
    """Check input headers for SEG-Y input to help determine geometry.

    This function reads in trace_qc_count headers and finds the unique cable values.
    The function then checks to make sure channel numbers for different cables do
    not overlap.

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
    for idx, cable in enumerate(unique_cables):
        min_val = cable_chan_min[idx]
        max_val = cable_chan_max[idx]
        for idx2, cable2 in enumerate(unique_cables):
            if cable2 == cable:
                continue

            if cable_chan_min[idx2] < max_val and cable_chan_max[idx2] > min_val:
                geom_type = StreamerShotGeometryType.A

    return unique_cables, cable_chan_min, cable_chan_max, geom_type


class GridOverrideCommand(ABC):
    """Abstract base class for grid override commands."""

    @property
    @abstractmethod
    def required_keys(self) -> set:
        """Get the set of required keys for the grid override command."""

    @abstractmethod
    def validate(self, index_headers: npt.NDArray, **kwargs) -> None:
        """Validate if this transform should run on the type of data."""

    @abstractmethod
    def transform(self, index_headers: npt.NDArray, **kwargs) -> npt.NDArray:
        """Perform the grid transform."""

    @property
    def name(self) -> str:
        """Convenience property to get the name of the command."""
        return self.__class__.__name__

    def check_required_keys(self, index_headers: npt.NDArray) -> None:
        """Check if all required keys are present in the index headers."""
        index_names = index_headers.keys()
        if not self.required_keys.issubset(index_names):
            raise GridOverrideKeysError(self.name, self.required_keys)


class AutoChannelWrap(GridOverrideCommand):
    """Automatically determine Streamer acquisition type."""

    required_keys = {"shot", "cable", "channel"}

    def validate(self, index_headers: npt.NDArray, **kwargs) -> None:
        """Validate if this transform should run on the type of data."""
        self.check_required_keys(index_headers)

        if "ChannelWrap" in kwargs:
            raise GridOverrideIncompatibleError(self.name, "ChannelWrap")

        if "CalculateCable" in kwargs:
            raise GridOverrideIncompatibleError(self.name, "CalculateCable")

    def transform(self, index_headers: npt.NDArray, **kwargs):
        """Perform the grid transform."""
        self.validate(index_headers, **kwargs)

        result = analyze_streamer_headers(index_headers)
        unique_cables, cable_chan_min, cable_chan_max, geom_type = result
        logger.info(f"Ingesting dataset as {geom_type.name}")

        # TODO: Add strict=True and remove noqa when min Python is 3.10
        for cable, chan_min, chan_max in zip(  # noqa: B905
            unique_cables, cable_chan_min, cable_chan_max
        ):
            logger.info(
                f"Cable: {cable} has min chan: {chan_min} and max chan: {chan_max}"
            )

        # This might be slow and potentially could be improved with a rewrite
        # to prevent so many lookups
        if geom_type == StreamerShotGeometryType.B:
            for idx, cable in enumerate(unique_cables):
                cable_idxs = np.where(index_headers["cable"][:] == cable)
                cc_min = cable_chan_min[idx]

                index_headers["channel"][cable_idxs] = (
                    index_headers["channel"][cable_idxs] - cc_min + 1
                )

        return index_headers


class ChannelWrap(GridOverrideCommand):
    """Wrap channels to start from one at cable boundaries."""

    required_keys = {"shot", "cable", "channel"}

    def validate(self, index_headers: npt.NDArray, **kwargs) -> None:
        """Validate if this transform should run on the type of data."""
        self.check_required_keys(index_headers)

        if "ChannelsPerCable" not in kwargs:
            raise GridOverrideMissingParameterError(self.name, "ChannelsPerCable")

        if "AutoCableChannel" in kwargs:
            raise GridOverrideIncompatibleError(self.name, "AutoCableChannel")

    def transform(self, index_headers: npt.NDArray, **kwargs) -> npt.NDArray:
        """Perform the grid transform."""
        self.validate(index_headers, **kwargs)

        channels_per_cable = kwargs["ChannelsPerCable"]
        index_headers["channel"] = (
            index_headers["channel"] - 1
        ) % channels_per_cable + 1

        return index_headers


class CalculateCable(GridOverrideCommand):
    """Calculate cable numbers from unwrapped channels."""

    required_keys = {"shot", "cable", "channel"}

    def validate(self, index_headers: npt.NDArray, **kwargs) -> None:
        """Validate if this transform should run on the type of data."""
        self.check_required_keys(index_headers)

        if "ChannelsPerCable" not in kwargs:
            raise GridOverrideMissingParameterError(self.name, "ChannelsPerCable")

        if "AutoCableChannel" in kwargs:
            raise GridOverrideIncompatibleError(self.name, "AutoCableChannel")

    def transform(self, index_headers, **kwargs):
        """Perform the grid transform."""
        channels_per_cable = kwargs["ChannelsPerCable"]
        index_headers["cable"] = (
            index_headers["channel"] - 1
        ) // channels_per_cable + 1

        return index_headers


class GridOverrider:
    """Executor for grid overrides.

    We support a certain type of grid overrides, and they have to be
    implemented following the ABC's in this module.

    This class applies the grid overrides if needed.
    """

    def __init__(self):
        """Define allowed overrides here."""
        self.commands = {
            "AutoChannelWrap": AutoChannelWrap(),
            "CalculateCable": CalculateCable(),
            "ChannelWrap": ChannelWrap(),
        }

    def run(
        self,
        index_headers: npt.NDArray,
        grid_overrides: dict[str, bool],
    ) -> npt.NDArray:
        """Run grid overrides and return result."""
        for override in grid_overrides:
            if override in self.commands:
                function = self.commands[override].transform
                index_headers = function(index_headers, grid_overrides=grid_overrides)
            else:
                raise GridOverrideUnknownError(override)

        return index_headers
