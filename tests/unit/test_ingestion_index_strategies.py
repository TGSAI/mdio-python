"""Tests for index strategies."""

import numpy as np
import pytest
from numpy import arange
from numpy import column_stack
from numpy import meshgrid

from mdio.ingestion.index_strategies import ChannelWrappingStrategy
from mdio.ingestion.index_strategies import CompositeStrategy
from mdio.ingestion.index_strategies import DuplicateHandlingStrategy
from mdio.ingestion.index_strategies import IndexStrategyFactory
from mdio.ingestion.index_strategies import NonBinnedStrategy
from mdio.ingestion.index_strategies import RegularGridStrategy
from mdio.segy.geometry import GridOverrides

SHOTS = arange(100, 104, dtype="int32")
CABLES = arange(11, 15, dtype="int32")
RECEIVERS = arange(1, 6, dtype="int32")


@pytest.fixture
def mock_streamer_headers():
    """Generate mocked streamer index headers."""
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


class TestRegularGridStrategy:
    """Tests for RegularGridStrategy."""

    def test_no_transformation(self, mock_streamer_headers) -> None:
        """Test that regular grid strategy doesn't transform headers."""
        strategy = RegularGridStrategy()

        result = strategy.transform_headers(mock_streamer_headers)

        # Headers should be unchanged
        assert result.dtype.names == mock_streamer_headers.dtype.names
        assert len(result) == len(mock_streamer_headers)

    def test_compute_dimensions(self, mock_streamer_headers) -> None:
        """Test dimension computation."""
        strategy = RegularGridStrategy()

        dim_names = ("shot_point", "cable", "channel")
        dimensions = strategy.compute_dimensions(mock_streamer_headers, dim_names)

        assert len(dimensions) == 3
        assert dimensions[0].name == "shot_point"
        assert dimensions[1].name == "cable"
        assert dimensions[2].name == "channel"
        np.testing.assert_array_equal(dimensions[0].coords, SHOTS)
        np.testing.assert_array_equal(dimensions[1].coords, CABLES)


class TestNonBinnedStrategy:
    """Tests for NonBinnedStrategy."""

    def test_transform_adds_trace_index(self, mock_streamer_headers) -> None:
        """Test that non-binned strategy adds trace index."""
        strategy = NonBinnedStrategy(chunksize=64)

        result = strategy.transform_headers(mock_streamer_headers)

        # Should have trace field added
        assert "trace" in result.dtype.names
        assert len(result) == len(mock_streamer_headers)

    def test_compute_dimensions_default_collapse(self, mock_streamer_headers) -> None:
        """Test dimension computation with default collapse (all but first)."""
        strategy = NonBinnedStrategy(chunksize=64)

        # Transform headers first
        transformed = strategy.transform_headers(mock_streamer_headers)

        # Compute dimensions (should replace cable and channel with trace)
        dim_names = ("shot_point", "cable", "channel")
        dimensions = strategy.compute_dimensions(transformed, dim_names)

        # Should have shot_point and trace
        assert len(dimensions) == 2
        assert dimensions[0].name == "shot_point"
        assert dimensions[1].name == "trace"

    def test_compute_dimensions_custom_collapse(self, mock_streamer_headers) -> None:
        """Test dimension computation with custom collapse dims."""
        strategy = NonBinnedStrategy(chunksize=128, collapse_dims=["channel"])

        # Transform headers first
        transformed = strategy.transform_headers(mock_streamer_headers)

        # Compute dimensions (should replace only channel with trace)
        dim_names = ("shot_point", "cable", "channel")
        dimensions = strategy.compute_dimensions(transformed, dim_names)

        # Should have shot_point, cable, and trace
        assert len(dimensions) == 3
        assert dimensions[0].name == "shot_point"
        assert dimensions[1].name == "cable"
        assert dimensions[2].name == "trace"


class TestDuplicateHandlingStrategy:
    """Tests for DuplicateHandlingStrategy."""

    def test_transform_adds_trace_index(self, mock_streamer_headers) -> None:
        """Test that duplicate handling adds trace index."""
        strategy = DuplicateHandlingStrategy()

        result = strategy.transform_headers(mock_streamer_headers)

        # Should have trace field added
        assert "trace" in result.dtype.names
        assert len(result) == len(mock_streamer_headers)

    def test_compute_dimensions(self, mock_streamer_headers) -> None:
        """Test dimension computation includes trace."""
        strategy = DuplicateHandlingStrategy()

        # Transform headers first
        transformed = strategy.transform_headers(mock_streamer_headers)

        # Compute dimensions
        dim_names = ("shot_point", "cable", "channel", "trace")
        dimensions = strategy.compute_dimensions(transformed, dim_names)

        # Should have all dimensions including trace
        assert len(dimensions) == 4
        assert dimensions[-1].name == "trace"


class TestChannelWrappingStrategy:
    """Tests for ChannelWrappingStrategy."""

    def test_transform_type_a(self) -> None:
        """Test channel wrapping for Type A (overlapping channels)."""
        # Create Type A headers (channels restart for each cable)
        headers = np.array(
            [
                (1, 1, 1),
                (1, 1, 2),
                (1, 2, 1),
                (1, 2, 2),
            ],
            dtype=[("shot_point", "int32"), ("cable", "int32"), ("channel", "int32")],
        )

        strategy = ChannelWrappingStrategy()
        result = strategy.transform_headers(headers)

        # Channels should remain unchanged for Type A
        np.testing.assert_array_equal(result["channel"], [1, 2, 1, 2])

    def test_transform_type_b(self) -> None:
        """Test channel wrapping for Type B (sequential channels)."""
        # Create Type B headers (channels sequential across cables)
        headers = np.array(
            [
                (1, 1, 1),
                (1, 1, 2),
                (1, 2, 3),
                (1, 2, 4),
            ],
            dtype=[("shot_point", "int32"), ("cable", "int32"), ("channel", "int32")],
        )

        strategy = ChannelWrappingStrategy()
        result = strategy.transform_headers(headers)

        # Channels should be wrapped to restart for each cable
        np.testing.assert_array_equal(result["channel"], [1, 2, 1, 2])


class TestCompositeStrategy:
    """Tests for CompositeStrategy."""

    def test_requires_strategies(self) -> None:
        """Test that composite strategy requires at least one strategy."""
        with pytest.raises(ValueError, match="at least one strategy"):
            CompositeStrategy([])

    def test_applies_strategies_in_order(self, mock_streamer_headers) -> None:
        """Test that strategies are applied in sequence."""
        # Create composite with multiple strategies
        strategies = [
            ChannelWrappingStrategy(),
            NonBinnedStrategy(chunksize=64),
        ]
        composite = CompositeStrategy(strategies)

        result = composite.transform_headers(mock_streamer_headers)

        # Should have trace field from NonBinned
        assert "trace" in result.dtype.names


class TestIndexStrategyFactory:
    """Tests for IndexStrategyFactory."""

    def test_no_overrides_returns_regular_grid(self) -> None:
        """Test that factory returns RegularGridStrategy with no overrides."""
        factory = IndexStrategyFactory()

        strategy = factory.create_strategy(grid_overrides=None)

        assert isinstance(strategy, RegularGridStrategy)

    def test_non_binned_override(self) -> None:
        """Test factory with non_binned override."""
        factory = IndexStrategyFactory()
        overrides = GridOverrides(non_binned=True, chunksize=64)

        strategy = factory.create_strategy(grid_overrides=overrides)

        assert isinstance(strategy, NonBinnedStrategy)
        assert strategy.chunksize == 64

    def test_has_duplicates_override(self) -> None:
        """Test factory with has_duplicates override."""
        factory = IndexStrategyFactory()
        overrides = GridOverrides(has_duplicates=True)

        strategy = factory.create_strategy(grid_overrides=overrides)

        assert isinstance(strategy, DuplicateHandlingStrategy)

    def test_auto_channel_wrap_override(self) -> None:
        """Test factory with auto_channel_wrap override."""
        factory = IndexStrategyFactory()
        overrides = GridOverrides(auto_channel_wrap=True)

        strategy = factory.create_strategy(grid_overrides=overrides)

        assert isinstance(strategy, ChannelWrappingStrategy)

    def test_composite_multiple_overrides(self) -> None:
        """Test factory creates composite for multiple overrides."""
        factory = IndexStrategyFactory()
        overrides = GridOverrides(auto_channel_wrap=True, non_binned=True, chunksize=64)

        strategy = factory.create_strategy(grid_overrides=overrides)

        # Should be composite with two strategies
        assert isinstance(strategy, CompositeStrategy)
        assert len(strategy.strategies) == 2
        assert isinstance(strategy.strategies[0], ChannelWrappingStrategy)
        assert isinstance(strategy.strategies[1], NonBinnedStrategy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
