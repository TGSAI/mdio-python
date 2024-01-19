"""Check grid overrides."""


from __future__ import annotations

from typing import Any

import numpy.typing as npt
import pytest
from numpy import arange
from numpy import column_stack
from numpy import meshgrid
from numpy import unique
from numpy.testing import assert_array_equal

from mdio.core import Dimension
from mdio.seismic.exceptions import GridOverrideIncompatibleError
from mdio.seismic.exceptions import GridOverrideMissingParameterError
from mdio.seismic.exceptions import GridOverrideUnknownError
from mdio.seismic.geometry import GridOverrider

SHOTS = arange(100, 104, dtype="int32")
CABLES = arange(11, 15, dtype="int32")
RECEIVERS = arange(1, 6, dtype="int32")


def run_override(
    grid_overrides: dict[str, Any],
    index_names: tuple[str, ...],
    headers: dict[str, npt.NDArray],
    chunksize: tuple[int, ...] | None = None,
) -> tuple[dict[str, Any], tuple[str], tuple[int]]:
    """Initialize and run overrider."""
    overrider = GridOverrider()
    return overrider.run(headers, index_names, grid_overrides, chunksize)


def get_dims(headers: dict[str, npt.NDArray]) -> list[Dimension]:
    """Get list of Dimensions from headers."""
    dims = []
    for index_name, index_coords in headers.items():
        dim_unique = unique(index_coords)
        dims.append(Dimension(coords=dim_unique, name=index_name))

    return dims


@pytest.fixture()
def mock_streamer_headers() -> dict[str, npt.NDArray]:
    """Generate dictionary of mocked streamer index headers."""
    grids = meshgrid(SHOTS, CABLES, RECEIVERS, indexing="ij")
    permutations = column_stack([grid.ravel() for grid in grids])

    # Make channel from receiver ids
    for shot in SHOTS:
        shot_mask = permutations[:, 0] == shot
        permutations[shot_mask, -1] = arange(1, len(CABLES) * len(RECEIVERS) + 1)

    return {
        "shot_point": permutations[:, 0],
        "cable": permutations[:, 1],
        "channel": permutations[:, 2],
    }


class TestAutoGridOverrides:
    """Check grid overrides works with auto indexing."""

    def test_duplicates(self, mock_streamer_headers: dict[str, npt.NDArray]) -> None:
        """Test the HasDuplicates Grid Override command."""
        index_names = ("shot", "cable")
        grid_overrides = {"HasDuplicates": True}

        # Remove channel header
        del mock_streamer_headers["channel"]
        chunksize = (4, 4, 8)

        new_headers, new_names, new_chunks = run_override(
            grid_overrides,
            index_names,
            mock_streamer_headers,
            chunksize,
        )

        assert new_names == ("shot", "cable", "trace")
        assert new_chunks == (4, 4, 1, 8)

        dims = get_dims(new_headers)

        assert_array_equal(dims[0].coords, SHOTS)
        assert_array_equal(dims[1].coords, CABLES)
        assert_array_equal(dims[2].coords, RECEIVERS)

    def test_non_binned(self, mock_streamer_headers: dict[str, npt.NDArray]) -> None:
        """Test the NonBinned Grid Override command."""
        index_names = ("shot", "cable")
        grid_overrides = {"NonBinned": True, "chunksize": 4}

        # Remove channel header
        del mock_streamer_headers["channel"]
        chunksize = (4, 4, 8)

        new_headers, new_names, new_chunks = run_override(
            grid_overrides,
            index_names,
            mock_streamer_headers,
            chunksize,
        )

        assert new_names == ("shot", "cable", "trace")
        assert new_chunks == (4, 4, 4, 8)

        dims = get_dims(new_headers)

        assert_array_equal(dims[0].coords, SHOTS)
        assert_array_equal(dims[1].coords, CABLES)
        assert_array_equal(dims[2].coords, RECEIVERS)


class TestStreamerGridOverrides:
    """Check grid overrides for shot data with streamer acquisition."""

    def test_channel_wrap(self, mock_streamer_headers: dict[str, npt.NDArray]) -> None:
        """Test the ChannelWrap command."""
        index_names = ("shot", "cable", "channel")
        grid_overrides = {"ChannelWrap": True, "ChannelsPerCable": len(RECEIVERS)}

        new_headers, new_names, new_chunks = run_override(
            grid_overrides, index_names, mock_streamer_headers
        )

        assert new_names == index_names
        assert new_chunks is None

        dims = get_dims(new_headers)

        assert_array_equal(dims[0].coords, SHOTS)
        assert_array_equal(dims[1].coords, CABLES)
        assert_array_equal(dims[2].coords, RECEIVERS)

    def test_calculate_cable(
        self,
        mock_streamer_headers: dict[str, npt.NDArray],
    ) -> None:
        """Test the CalculateCable command."""
        index_names = ("shot", "cable", "channel")
        grid_overrides = {"CalculateCable": True, "ChannelsPerCable": len(RECEIVERS)}

        new_headers, new_names, new_chunks = run_override(
            grid_overrides, index_names, mock_streamer_headers
        )

        assert new_names == index_names
        assert new_chunks is None

        dims = get_dims(new_headers)

        # We need channels because unwrap isn't done here
        channels = unique(mock_streamer_headers["channel"])

        # We reset the cables to start from 1.
        cables = arange(1, len(CABLES) + 1, dtype="uint32")

        assert_array_equal(dims[0].coords, SHOTS)
        assert_array_equal(dims[1].coords, cables)
        assert_array_equal(dims[2].coords, channels)

    def test_wrap_and_calc_cable(
        self,
        mock_streamer_headers: dict[str, npt.NDArray],
    ) -> None:
        """Test the combined ChannelWrap and CalculateCable commands."""
        index_names = ("shot", "cable", "channel")
        grid_overrides = {
            "CalculateCable": True,
            "ChannelWrap": True,
            "ChannelsPerCable": len(RECEIVERS),
        }

        new_headers, new_names, new_chunks = run_override(
            grid_overrides, index_names, mock_streamer_headers
        )

        assert new_names == index_names
        assert new_chunks is None

        dims = get_dims(new_headers)
        # We reset the cables to start from 1.
        cables = arange(1, len(CABLES) + 1, dtype="uint32")

        assert_array_equal(dims[0].coords, SHOTS)
        assert_array_equal(dims[1].coords, cables)
        assert_array_equal(dims[2].coords, RECEIVERS)

    def test_missing_param(self, mock_streamer_headers: dict[str, npt.NDArray]) -> None:
        """Test missing parameters for the commands."""
        index_names = ("shot", "cable", "channel")
        chunksize = None
        overrider = GridOverrider()

        with pytest.raises(GridOverrideMissingParameterError):
            overrider.run(
                mock_streamer_headers, index_names, {"ChannelWrap": True}, chunksize
            )

        with pytest.raises(GridOverrideMissingParameterError):
            overrider.run(
                mock_streamer_headers, index_names, {"CalculateCable": True}, chunksize
            )

    def test_incompatible_overrides(
        self,
        mock_streamer_headers: dict[str, npt.NDArray],
    ) -> None:
        """Test commands that can't be run together."""
        index_names = ("shot", "cable", "channel")
        chunksize = None
        overrider = GridOverrider()

        grid_overrides = {"ChannelWrap": True, "AutoChannelWrap": True}
        with pytest.raises(GridOverrideIncompatibleError):
            overrider.run(mock_streamer_headers, index_names, grid_overrides, chunksize)

        grid_overrides = {"CalculateCable": True, "AutoChannelWrap": True}
        with pytest.raises(GridOverrideIncompatibleError):
            overrider.run(mock_streamer_headers, index_names, grid_overrides, chunksize)

    def test_unknown_override(
        self,
        mock_streamer_headers: dict[str, npt.NDArray],
    ) -> None:
        """Test exception if user provides a command that's not allowed."""
        index_names = ("shot", "cable", "channel")
        chunksize = None
        overrider = GridOverrider()
        with pytest.raises(GridOverrideUnknownError):
            overrider.run(
                mock_streamer_headers, index_names, {"WrongCommand": True}, chunksize
            )
