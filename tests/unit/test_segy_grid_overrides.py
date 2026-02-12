"""Check grid overrides."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from numpy import arange
from numpy import column_stack
from numpy import meshgrid
from numpy import unique
from numpy.testing import assert_array_equal

from mdio.core import Dimension
from mdio.segy.exceptions import GridOverrideUnknownError
from mdio.segy.geometry import GridOverrider

if TYPE_CHECKING:
    from mdio.builder.templates.base import AbstractDatasetTemplate

SHOTS = arange(100, 104, dtype="int32")
CABLES = arange(11, 15, dtype="int32")
RECEIVERS = arange(1, 6, dtype="int32")


def run_override(
    grid_overrides: dict[str, Any],
    index_names: tuple[str, ...],
    headers: npt.NDArray,
    chunksize: tuple[int, ...] | None = None,
    template: AbstractDatasetTemplate | None = None,
) -> tuple[dict[str, Any], tuple[str], tuple[int]]:
    """Initialize and run overrider."""
    overrider = GridOverrider()
    return overrider.run(headers, index_names, grid_overrides, chunksize, template=template)


def get_dims(headers: npt.NDArray) -> list[Dimension]:
    """Get list of Dimensions from headers."""
    dims = []
    for index_name in headers.dtype.names:
        index_coords = headers[index_name]
        dim_unique = unique(index_coords)
        dims.append(Dimension(coords=dim_unique, name=index_name))

    return dims


@pytest.fixture
def mock_streamer_headers() -> npt.NDArray:
    """Generate dictionary of mocked streamer index headers."""
    grids = meshgrid(SHOTS, CABLES, RECEIVERS, indexing="ij")
    permutations = column_stack([grid.ravel() for grid in grids])

    # Make channel from receiver ids
    for shot in SHOTS:
        shot_mask = permutations[:, 0] == shot
        permutations[shot_mask, -1] = arange(1, len(CABLES) * len(RECEIVERS) + 1)

    hdr_dtype = np.dtype(
        {
            "names": ["shot_point", "cable", "channel"],
            "formats": ["int32", "int32", "int32"],
        }
    )

    n_traces = permutations.shape[0]
    result = np.ndarray(dtype=hdr_dtype, shape=n_traces)

    result["shot_point"] = permutations[:, 0]
    result["cable"] = permutations[:, 1]
    result["channel"] = permutations[:, 2]

    return result


class TestAutoGridOverrides:
    """Check grid overrides works with auto indexing."""

    def test_duplicates(self, mock_streamer_headers: dict[str, npt.NDArray]) -> None:
        """Test the HasDuplicates Grid Override command."""
        index_names = ("shot_point", "cable")
        grid_overrides = {"HasDuplicates": True}

        # Remove channel header
        streamer_headers = mock_streamer_headers[list(index_names)]
        chunksize = (4, 4, 8)

        new_headers, new_names, new_chunks = run_override(
            grid_overrides,
            index_names,
            streamer_headers,
            chunksize,
        )

        assert new_names == ("shot_point", "cable", "trace")
        assert new_chunks == (4, 4, 1, 8)

        dims = get_dims(new_headers)

        assert_array_equal(dims[0].coords, SHOTS)
        assert_array_equal(dims[1].coords, CABLES)
        assert_array_equal(dims[2].coords, RECEIVERS)

    def test_non_binned(self, mock_streamer_headers: dict[str, npt.NDArray]) -> None:
        """Test the NonBinned Grid Override command."""
        index_names = ("shot_point", "cable")
        grid_overrides = {"NonBinned": True, "chunksize": 4, "non_binned_dims": ["channel"]}

        # Keep channel header for non-binned processing
        streamer_headers = mock_streamer_headers
        chunksize = (4, 4, 8)

        new_headers, new_names, new_chunks = run_override(
            grid_overrides,
            index_names,
            streamer_headers,
            chunksize,
        )

        assert new_names == ("shot_point", "cable", "trace")
        assert new_chunks == (4, 4, 4, 8)

        dims = get_dims(new_headers)

        assert_array_equal(dims[0].coords, SHOTS)
        assert_array_equal(dims[1].coords, CABLES)
        # Trace coords are the unique channel values (1-20)
        expected_trace_coords = np.arange(1, 21, dtype="int32")
        assert_array_equal(dims[2].coords, expected_trace_coords)


class TestStreamerGridOverrides:
    """Check grid overrides for shot data with streamer acquisition."""

    def test_unknown_override(
        self,
        mock_streamer_headers: dict[str, npt.NDArray],
    ) -> None:
        """Test exception if user provides a command that's not allowed."""
        index_names = ("shot_point", "cable", "channel")
        chunksize = None
        overrider = GridOverrider()
        with pytest.raises(GridOverrideUnknownError):
            overrider.run(mock_streamer_headers, index_names, {"WrongCommand": True}, chunksize)


# OBN test fixtures and tests
OBN_RECEIVERS = arange(1, 501, dtype="int32")[:10]  # Use subset for tests
OBN_SHOT_LINES = arange(1, 11, dtype="int32")[:3]
OBN_GUNS = arange(1, 3, dtype="int8")
OBN_SHOT_POINTS = arange(1, 201, dtype="int32")[:5]


class TestObnGridOverrides:
    """Check grid overrides for OBN (Ocean Bottom Node) data.

    Note: The synthetic component behavior (when SEG-Y spec doesn't have a component field)
    is handled by the template's internal `_synthetic_defaults` attribute in `utilities.get_grid_plan()`,
    not by a grid override. See integration test `test_import_obn_synthetic_component` for
    full flow coverage. These unit tests focus on grid override functionality.
    """

    def test_calculate_shot_index_obn(self) -> None:
        """Test CalculateShotIndex calculates shot_index from interleaved shot_points for OBN.

        CalculateShotIndex is specific to OBN templates and uses shot_line
        without requiring cable/channel headers.
        """
        from mdio.builder.template_registry import TemplateRegistry  # noqa: PLC0415

        # Create headers with interleaved shot points (odd for gun 1, even for gun 2)
        # Simulating Type B geometry where shot points are gun1: 1,3,5... gun2: 2,4,6...
        receivers = arange(1, 4, dtype="int32")
        shot_lines = arange(1, 3, dtype="int32")
        guns = arange(1, 3, dtype="int8")  # 2 guns
        # Shot points are interleaved: gun1 gets 1,3,5 and gun2 gets 2,4,6
        shot_points_gun1 = arange(1, 7, 2, dtype="int32")  # 1, 3, 5
        shot_points_gun2 = arange(2, 8, 2, dtype="int32")  # 2, 4, 6

        # Build headers manually for interleaved geometry
        records = [
            (receiver, shot_line, gun, shot_point)
            for shot_line in shot_lines
            for receiver in receivers
            for gun in guns
            for shot_point in (shot_points_gun1 if gun == 1 else shot_points_gun2)
        ]

        hdr_dtype = np.dtype(
            {
                "names": ["receiver", "shot_line", "gun", "shot_point"],
                "formats": ["int32", "int32", "int8", "int32"],
            }
        )
        headers = np.array(records, dtype=hdr_dtype)

        index_names = ("receiver", "shot_line", "gun", "shot_point")
        grid_overrides = {"CalculateShotIndex": True}
        chunksize = (8, 1, 2, 8, 4096)

        # Use OBN template for CalculateShotIndex
        obn_template = TemplateRegistry().get("ObnReceiverGathers3D")

        new_headers, new_names, new_chunks = run_override(
            grid_overrides,
            index_names,
            headers,
            chunksize,
            template=obn_template,
        )

        # shot_index should be added
        assert "shot_index" in new_headers.dtype.names

        # Verify shot_index is calculated correctly
        # floor(shot_point / num_guns) - min gives:
        # gun1: floor(1/2)=0, floor(3/2)=1, floor(5/2)=2 -> 0, 1, 2
        # gun2: floor(2/2)=1, floor(4/2)=2, floor(6/2)=3 -> 1, 2, 3
        # Combined unique values are [0, 1, 2, 3] (since min is 0)
        unique_shot_index = np.unique(new_headers["shot_index"])
        assert_array_equal(unique_shot_index, [0, 1, 2, 3])

        # Verify per-gun shot_index values
        gun1_mask = new_headers["gun"] == 1
        gun2_mask = new_headers["gun"] == 2
        gun1_shot_indices = np.unique(new_headers["shot_index"][gun1_mask])
        gun2_shot_indices = np.unique(new_headers["shot_index"][gun2_mask])
        assert_array_equal(gun1_shot_indices, [0, 1, 2])
        assert_array_equal(gun2_shot_indices, [1, 2, 3])
