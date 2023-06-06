"""Check grid overrides."""


from __future__ import annotations

import numpy.typing as npt
import pytest
from numpy import arange
from numpy import column_stack
from numpy import meshgrid
from numpy import unique
from numpy.testing import assert_array_equal

from mdio.core import Dimension
from mdio.segy.exceptions import GridOverrideIncompatibleError
from mdio.segy.exceptions import GridOverrideMissingParameterError
from mdio.segy.exceptions import GridOverrideUnknownError
from mdio.segy.geometry import GridOverrider


SHOTS = arange(100, 104, dtype="int32")
CABLES = arange(11, 15, dtype="int32")
RECEIVERS = arange(1, 6, dtype="int32")


@pytest.fixture
def mock_streamer_headers() -> dict[str, npt.NDArray]:
    """Generate dictionary of mocked streamer index headers."""
    grids = meshgrid(SHOTS, CABLES, RECEIVERS, indexing="ij")
    permutations = column_stack([grid.ravel() for grid in grids])

    # Make channel from receiver ids
    for shot in SHOTS:
        shot_mask = permutations[:, 0] == shot
        permutations[shot_mask, -1] = arange(1, len(CABLES) * len(RECEIVERS) + 1)

    result = dict(
        shot=permutations[:, 0],
        cable=permutations[:, 1],
        channel=permutations[:, 2],
    )

    return result


class TestStreamerGridOverrides:
    """Check grid overrides for shot data with streamer acquisition."""

    def test_channel_wrap(self, mock_streamer_headers: npt.NDArray) -> None:
        """Test the ChannelWrap command."""
        grid_overrides = {"ChannelWrap": True, "ChannelsPerCable": len(RECEIVERS)}

        overrider = GridOverrider()
        results = overrider.run(mock_streamer_headers, grid_overrides)

        dims = []
        for index_name, index_coords in results.items():
            dim_unique = unique(index_coords)
            dims.append(Dimension(coords=dim_unique, name=index_name))

        assert_array_equal(dims[0], SHOTS)
        assert_array_equal(dims[1], CABLES)
        assert_array_equal(dims[2], RECEIVERS)

    def test_calculate_cable(self, mock_streamer_headers: npt.NDArray) -> None:
        """Test the CalculateCable command."""
        grid_overrides = {
            "CalculateCable": True,
            "ChannelsPerCable": len(RECEIVERS),
        }

        overrider = GridOverrider()
        results = overrider.run(mock_streamer_headers, grid_overrides)

        dims = []
        for index_name, index_coords in results.items():
            dim_unique = unique(index_coords)
            dims.append(Dimension(coords=dim_unique, name=index_name))

        # We need channels because unwrap isn't done here
        channels = unique(mock_streamer_headers["channel"])

        # We reset the cables to start from 1.
        cables = arange(1, len(CABLES) + 1, dtype="uint32")

        assert_array_equal(dims[0], SHOTS)
        assert_array_equal(dims[1], cables)
        assert_array_equal(dims[2], channels)

    def test_wrap_and_calc_cable(self, mock_streamer_headers: npt.NDArray) -> None:
        """Test the combined ChannelWrap and CalculateCable commands."""
        grid_overrides = {
            "CalculateCable": True,
            "ChannelWrap": True,
            "ChannelsPerCable": len(RECEIVERS),
        }

        overrider = GridOverrider()
        results = overrider.run(mock_streamer_headers, grid_overrides)

        dims = []
        for index_name, index_coords in results.items():
            dim_unique = unique(index_coords)
            dims.append(Dimension(coords=dim_unique, name=index_name))

        # We reset the cables to start from 1.
        cables = arange(1, len(CABLES) + 1, dtype="uint32")

        assert_array_equal(dims[0], SHOTS)
        assert_array_equal(dims[1], cables)
        assert_array_equal(dims[2], RECEIVERS)

    def test_missing_param(self, mock_streamer_headers: npt.NDArray) -> None:
        """Test missing parameters for the commands."""
        overrider = GridOverrider()
        with pytest.raises(GridOverrideMissingParameterError):
            overrider.run(mock_streamer_headers, {"ChannelWrap": True})

        with pytest.raises(GridOverrideMissingParameterError):
            overrider.run(mock_streamer_headers, {"CalculateCable": True})

    def test_incompatible_overrides(self, mock_streamer_headers: npt.NDArray) -> None:
        """Test commands that can't be run together."""
        overrider = GridOverrider()
        with pytest.raises(GridOverrideIncompatibleError):
            grid_overrides = {"ChannelWrap": True, "AutoChannelWrap": True}
            overrider.run(mock_streamer_headers, grid_overrides)

        with pytest.raises(GridOverrideIncompatibleError):
            grid_overrides = {"CalculateCable": True, "AutoChannelWrap": True}
            overrider.run(mock_streamer_headers, grid_overrides)

    def test_unknown_override(self, mock_streamer_headers: npt.NDArray) -> None:
        """Test exception if user provides a command that's not allowed."""
        overrider = GridOverrider()
        with pytest.raises(GridOverrideUnknownError):
            overrider.run(mock_streamer_headers, {"WrongCommand": True})
