"""Index Strategy System for MDIO Ingestion.

This module implements the strategy pattern for transforming SEG-Y headers
into indexable dimensions. Each strategy handles a specific type of grid
override or indexing pattern.
"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib import recfunctions as rfn

from mdio.core import Dimension
from mdio.ingestion.header_analysis import StreamerShotGeometryType
from mdio.ingestion.header_analysis import analyze_non_indexed_headers
from mdio.ingestion.header_analysis import analyze_saillines_for_guns
from mdio.ingestion.header_analysis import analyze_streamer_headers

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from segy.arrays import HeaderArray

    from mdio.segy.geometry import GridOverrides

logger = logging.getLogger(__name__)


class IndexStrategy(ABC):
    """Base class for indexing strategies.

    Each strategy knows how to:
    1. Transform headers (add/modify header fields)
    2. Compute dimensions from the transformed headers
    """

    @abstractmethod
    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Transform headers for indexing.

        Args:
            headers: Input header array

        Returns:
            Transformed header array (may have additional fields)
        """

    @abstractmethod
    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions from headers.

        Args:
            headers: Transformed header array
            dim_names: Expected dimension names from schema

        Returns:
            List of Dimension objects
        """

    @property
    def name(self) -> str:
        """Return strategy name."""
        return self.__class__.__name__


class RegularGridStrategy(IndexStrategy):
    """Standard grid indexing without transformations.

    This is the default strategy when no grid overrides are specified.
    Headers are used as-is to build dimensions.
    """

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """No transformation needed for regular grids."""
        return headers

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions from header values."""
        dimensions = []

        for dim_name in dim_names:
            if dim_name in headers.dtype.names:
                coords = np.unique(headers[dim_name])
                dimensions.append(Dimension(coords=coords, name=dim_name))

        return dimensions


class NonBinnedStrategy(IndexStrategy):
    """Sequential trace indexing for non-binned data.

    Creates a "trace" dimension by sequentially numbering traces,
    replacing specified spatial dimensions.
    """

    def __init__(self, chunksize: int, collapse_dims: list[str] | None = None):
        """Initialize non-binned strategy.

        Args:
            chunksize: Chunk size for the trace dimension
            collapse_dims: Dimension names to collapse into trace.
                If None, inferred from dim_names in compute_dimensions.
        """
        self.chunksize = chunksize
        self.collapse_dims = collapse_dims or []

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Add sequential trace index to headers.

        For non-binned data, traces are simply numbered sequentially 0, 1, 2, ...
        This is different from DuplicateHandling which uses hierarchical indexing.
        """
        # Add simple sequential trace index
        trace_idx = np.arange(len(headers), dtype=np.int32)
        return rfn.append_fields(headers, "trace", trace_idx, usemask=False)

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions with trace replacing collapsed dims."""
        dimensions = []

        # If collapse_dims not specified, collapse all but first spatial dim
        if not self.collapse_dims:
            # Get all dims that are in headers (these are spatial, vertical is synthetic)
            spatial_dims = [d for d in dim_names if d in headers.dtype.names]
            # Collapse all but the first spatial dimension
            if len(spatial_dims) > 1:
                self.collapse_dims = spatial_dims[1:]

        # Track if we've added trace dimension
        trace_added = False

        for dim_name in dim_names:
            # Skip collapsed dimensions
            if dim_name in self.collapse_dims:
                # Add trace dimension only once when we encounter first collapsed dim
                if not trace_added and "trace" in headers.dtype.names:
                    coords = np.unique(headers["trace"])
                    dimensions.append(Dimension(coords=coords, name="trace"))
                    trace_added = True
                continue

            # Handle trace dimension specially if explicitly in dim_names
            if dim_name == "trace" and not trace_added:
                coords = np.unique(headers["trace"])
                dimensions.append(Dimension(coords=coords, name="trace"))
                trace_added = True
            elif dim_name in headers.dtype.names:
                coords = np.unique(headers[dim_name])
                dimensions.append(Dimension(coords=coords, name=dim_name))

        return dimensions


class DuplicateHandlingStrategy(IndexStrategy):
    """Handle duplicate indices by adding trace dimension.

    Similar to NonBinned but uses a fixed chunksize of 1 and doesn't
    collapse other dimensions.
    """

    def __init__(self, dtype: DTypeLike = np.int16):
        """Initialize duplicate handling strategy.

        Args:
            dtype: Data type for trace index
        """
        self.dtype = dtype

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Add trace index for duplicates."""
        return analyze_non_indexed_headers(headers, dtype=self.dtype)

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions including trace for duplicates."""
        dimensions = []

        for dim_name in dim_names:
            if dim_name in headers.dtype.names:
                coords = np.unique(headers[dim_name])
                dimensions.append(Dimension(coords=coords, name=dim_name))

        return dimensions


class ChannelWrappingStrategy(IndexStrategy):
    """Handle streamer acquisition channel wrapping (Type A/B).

    Analyzes channel numbering across cables and adjusts for Type B
    (sequential numbering) to Type A (per-cable numbering).
    """

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Adjust channel numbers based on geometry type."""
        result = analyze_streamer_headers(headers)
        unique_cables, cable_chan_min, cable_chan_max, geom_type = result

        logger.info("Ingesting dataset as %s", geom_type.name)

        for cable, chan_min, chan_max in zip(unique_cables, cable_chan_min, cable_chan_max, strict=True):
            logger.info("Cable: %s has min chan: %s and max chan: %s", cable, chan_min, chan_max)

        # Transform Type B to Type A
        if geom_type == StreamerShotGeometryType.B:
            for idx, cable in enumerate(unique_cables):
                cable_idxs = np.where(headers["cable"][:] == cable)
                cc_min = cable_chan_min[idx]
                headers["channel"][cable_idxs] = headers["channel"][cable_idxs] - cc_min + 1

        return headers

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions after channel wrapping."""
        dimensions = []

        for dim_name in dim_names:
            if dim_name in headers.dtype.names:
                coords = np.unique(headers[dim_name])
                dimensions.append(Dimension(coords=coords, name=dim_name))

        return dimensions


class ShotWrappingStrategy(IndexStrategy):
    """Handle multi-gun acquisition shot wrapping.

    Creates shot_index from shot_point for Type B geometry where
    shot points are numbered uniquely across guns.
    """

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Create shot_index for multi-gun acquisition."""
        result = analyze_saillines_for_guns(headers)
        unique_sail_lines, unique_guns_in_sail_line, geom_type = result

        logger.info("Ingesting dataset as shot type: %s", geom_type.name)

        max_num_guns = 1
        for sail_line in unique_sail_lines:
            logger.info("sail_line: %s has guns: %s", sail_line, unique_guns_in_sail_line[str(sail_line)])
            num_guns = len(unique_guns_in_sail_line[str(sail_line)])
            max_num_guns = max(num_guns, max_num_guns)

        # Transform Type B to add shot_index
        if geom_type.name == "B":
            shot_index = np.empty(len(headers), dtype="uint32")
            headers = rfn.append_fields(headers.base, "shot_index", shot_index)

            for sail_line in unique_sail_lines:
                sail_line_idxs = np.where(headers["sail_line"][:] == sail_line)
                headers["shot_index"][sail_line_idxs] = np.floor(headers["shot_point"][sail_line_idxs] / max_num_guns)
                # Make shot index zero-based PER sail line
                headers["shot_index"][sail_line_idxs] -= headers["shot_index"][sail_line_idxs].min()

        return headers

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions including shot_index if created."""
        dimensions = []

        for dim_name in dim_names:
            if dim_name in headers.dtype.names:
                coords = np.unique(headers[dim_name])
                dimensions.append(Dimension(coords=coords, name=dim_name))

        return dimensions


class CompositeStrategy(IndexStrategy):
    """Combines multiple strategies in sequence.

    Strategies are applied in the order provided. Each strategy's
    header transformation feeds into the next strategy.
    """

    def __init__(self, strategies: list[IndexStrategy]):
        """Initialize composite strategy.

        Args:
            strategies: List of strategies to apply in order
        """
        if not strategies:
            raise ValueError("CompositeStrategy requires at least one strategy")
        self.strategies = strategies

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Apply all strategy transformations in sequence."""
        result = headers
        for strategy in self.strategies:
            logger.debug("Applying strategy: %s", strategy.name)
            result = strategy.transform_headers(result)
        return result

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions using the last strategy's logic.

        Note: This assumes the last strategy knows about all transformations.
        For more complex cases, might need to coordinate between strategies.
        """
        return self.strategies[-1].compute_dimensions(headers, dim_names)


class IndexStrategyFactory:
    """Factory for creating index strategies from grid overrides.

    This factory encapsulates the logic of which strategies to use
    based on the grid override configuration.
    """

    def create_strategy(self, grid_overrides: GridOverrides | None = None) -> IndexStrategy:
        """Create appropriate index strategy from grid overrides.

        Args:
            grid_overrides: Optional grid override configuration

        Returns:
            IndexStrategy (may be composite if multiple overrides)
        """
        if grid_overrides is None or not grid_overrides:
            return RegularGridStrategy()

        strategies = []

        # Add channel wrapping if requested
        if grid_overrides.auto_channel_wrap:
            strategies.append(ChannelWrappingStrategy())

        # Add shot wrapping if requested
        if grid_overrides.auto_shot_wrap:
            strategies.append(ShotWrappingStrategy())

        # Add non-binned or duplicate handling (mutually exclusive)
        if grid_overrides.non_binned:
            strategies.append(
                NonBinnedStrategy(
                    chunksize=grid_overrides.chunksize,
                    collapse_dims=grid_overrides.replace_dims,
                )
            )
        elif grid_overrides.has_duplicates:
            strategies.append(DuplicateHandlingStrategy())

        # If no strategies added, use regular grid
        if not strategies:
            return RegularGridStrategy()

        # If single strategy, return it directly
        if len(strategies) == 1:
            return strategies[0]

        # Otherwise, return composite
        return CompositeStrategy(strategies)
