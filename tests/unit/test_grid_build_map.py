"""Tests for Grid.build_map batching and derived live mask."""

from unittest.mock import patch

import numpy as np

from mdio.core.grid import Grid
from tests.unit.ingestion.testing_helpers import make_grid_with_map

_DIMS = [
    ("inline", np.arange(10, dtype=np.int32)),
    ("crossline", np.arange(10, dtype=np.int32)),
    ("sample", np.arange(4, dtype=np.int32)),
]


class TestBuildMap:
    """Populate the trace map and live mask from header indices."""

    def test_live_mask_matches_map_across_batches_and_chunks(self) -> None:
        """Mask is `map != fill` across multi-batch vindex and a ragged chunk grid."""
        live = [(0, 0), (9, 9)]
        with (
            patch.object(Grid, "_TARGET_MEMORY_PER_BATCH", 64),
            patch.object(Grid, "_INTERNAL_CHUNK_SIZE_TARGET", 48),
        ):
            grid = make_grid_with_map(_DIMS, live)
            assert grid.map.cdata_shape == (3, 3)

        fill = grid.map.fill_value
        np.testing.assert_array_equal(np.asarray(grid.live_mask[:]), np.asarray(grid.map[:]) != fill)
        assert int(np.sum(grid.live_mask)) == len(live)
        for ordinal, (inline, crossline) in enumerate(live):
            assert grid.map[inline, crossline] == ordinal
            assert bool(grid.live_mask[inline, crossline]) is True
        assert bool(grid.live_mask[0, 1]) is False
