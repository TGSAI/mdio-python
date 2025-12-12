"""Check grid overrides."""

from __future__ import annotations

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
from mdio.ingestion import DuplicateHandlingStrategy
from mdio.ingestion import IndexStrategyFactory
from mdio.ingestion import NonBinnedStrategy
from mdio.ingestion import SchemaResolver
from mdio.segy.geometry import GridOverrides

SHOTS = arange(100, 104, dtype="int32")
CABLES = arange(11, 15, dtype="int32")
RECEIVERS = arange(1, 6, dtype="int32")


def run_override_strategies(
    grid_overrides: GridOverrides,
    headers: npt.NDArray,
) -> tuple[npt.NDArray, tuple[str, ...]]:
    """Run index strategies on headers."""
    # Create strategy from overrides
    factory = IndexStrategyFactory()
    strategy = factory.create_strategy(grid_overrides=grid_overrides)

    # Transform headers
    transformed_headers = strategy.transform_headers(headers)

    # Get dimension names
    dim_names = tuple(transformed_headers.dtype.names)

    return transformed_headers, dim_names


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
        """Test the HasDuplicates Grid Override."""
        # Remove channel header
        streamer_headers = mock_streamer_headers[["shot_point", "cable"]]

        grid_overrides = GridOverrides(has_duplicates=True)

        new_headers, new_names = run_override_strategies(grid_overrides, streamer_headers)

        assert new_names == ("shot_point", "cable", "trace")

        dims = get_dims(new_headers)

        assert_array_equal(dims[0].coords, SHOTS)
        assert_array_equal(dims[1].coords, CABLES)
        assert_array_equal(dims[2].coords, RECEIVERS)

    def test_non_binned(self, mock_streamer_headers: dict[str, npt.NDArray]) -> None:
        """Test the NonBinned Grid Override."""
        # Remove channel header
        streamer_headers = mock_streamer_headers[["shot_point", "cable"]]

        grid_overrides = GridOverrides(non_binned=True, chunksize=4)

        new_headers, new_names = run_override_strategies(grid_overrides, streamer_headers)

        # Adds sequential trace index
        assert "trace" in new_names
        assert "shot_point" in new_names
        assert "cable" in new_names

    def test_non_binned_with_replace_dims(self, mock_streamer_headers: dict[str, npt.NDArray]) -> None:
        """Test the NonBinned Grid Override with replace_dims parameter."""
        grid_overrides = GridOverrides(
            non_binned=True,
            chunksize=8,
            replace_dims=["cable", "channel"]
        )

        new_headers, new_names = run_override_strategies(grid_overrides, mock_streamer_headers)

        # Should keep shot_point and add trace, but header still has all fields
        # The dimension collapsing happens at schema resolution level
        assert "trace" in new_names
        assert "shot_point" in new_names


class TestStreamerGridOverrides:
    """Check grid overrides for shot data with streamer acquisition."""

    def test_unknown_override(
        self,
        mock_streamer_headers: dict[str, npt.NDArray],
    ) -> None:
        """Test exception if user provides invalid grid override configuration."""
        # Pydantic will catch unknown parameters at creation time
        with pytest.raises(Exception):  # Pydantic ValidationError
            GridOverrides(WrongCommand=True)


class TestTemplateTransformations:
    """Test template transformation methods for grid overrides using SchemaResolver."""

    def test_duplicate_index_template_transform(self) -> None:
        """Test HasDuplicates transforms template dimensions correctly via SchemaResolver."""
        from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate

        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        assert template.dimension_names == ("shot_point", "cable", "channel", "time")
        assert template.full_chunk_shape == (8, 1, 128, 2048)

        resolver = SchemaResolver()
        grid_overrides = GridOverrides(has_duplicates=True)
        schema = resolver.resolve(template, grid_overrides=grid_overrides)

        # Should add trace dimension before time
        dim_names = [d.name for d in schema.dimensions]
        assert dim_names == ["shot_point", "cable", "channel", "trace", "time"]

        # Chunk shape should have 1 for trace dimension
        expected_chunks = (8, 1, 128, 1, 2048)
        assert schema.chunk_shape == expected_chunks

    def test_non_binned_template_transform_default(self) -> None:
        """Test NonBinned with default replace_dims (all but first spatial dim) via SchemaResolver."""
        from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate

        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        assert template.dimension_names == ("shot_point", "cable", "channel", "time")
        assert template.full_chunk_shape == (8, 1, 128, 2048)

        resolver = SchemaResolver()
        grid_overrides = GridOverrides(non_binned=True, chunksize=64)
        schema = resolver.resolve(template, grid_overrides=grid_overrides)

        # Should replace cable and channel with trace (default behavior)
        dim_names = [d.name for d in schema.dimensions]
        assert dim_names == ["shot_point", "trace", "time"]

        expected_chunks = (8, 64, 2048)
        assert schema.chunk_shape == expected_chunks

    def test_non_binned_template_transform_with_replace_dims(self) -> None:
        """Test NonBinned with explicit replace_dims via SchemaResolver."""
        from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate

        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        assert template.dimension_names == ("shot_point", "cable", "channel", "time")

        resolver = SchemaResolver()
        grid_overrides = GridOverrides(non_binned=True, chunksize=128, replace_dims=["channel"])
        schema = resolver.resolve(template, grid_overrides=grid_overrides)

        # Should replace only channel with trace
        dim_names = [d.name for d in schema.dimensions]
        assert dim_names == ["shot_point", "cable", "trace", "time"]

        expected_chunks = (8, 1, 128, 2048)
        assert schema.chunk_shape == expected_chunks

    def test_non_binned_template_transform_all_dims(self) -> None:
        """Test NonBinned replacing all spatial dimensions via SchemaResolver."""
        from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate

        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")

        resolver = SchemaResolver()
        grid_overrides = GridOverrides(
            non_binned=True,
            chunksize=256,
            replace_dims=["shot_point", "cable", "channel"]
        )
        schema = resolver.resolve(template, grid_overrides=grid_overrides)

        # Should replace all spatial dims with trace
        dim_names = [d.name for d in schema.dimensions]
        assert dim_names == ["trace", "time"]

        expected_chunks = (256, 2048)
        assert schema.chunk_shape == expected_chunks

    def test_non_binned_coordinate_dimension_update(self) -> None:
        """Test that coordinate dimensions are updated when NonBinned collapses dimensions."""
        from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate

        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")

        resolver = SchemaResolver()
        # Default replace_dims will replace cable and channel
        grid_overrides = GridOverrides(non_binned=True, chunksize=64)
        schema = resolver.resolve(template, grid_overrides=grid_overrides)

        # Check dimension transformation
        dim_names = [d.name for d in schema.dimensions]
        assert dim_names == ["shot_point", "trace", "time"]

        # Check coordinate dimension updates
        coord_dims = {coord.name: coord.dimensions for coord in schema.coordinates}

        # Coordinates that didn't reference collapsed dims should stay the same
        assert coord_dims["gun"] == ("shot_point",)
        assert coord_dims["source_coord_x"] == ("shot_point",)
        assert coord_dims["source_coord_y"] == ("shot_point",)

        # Coordinates that referenced collapsed dims should have trace appended
        assert coord_dims["group_coord_x"] == ("shot_point", "trace")
        assert coord_dims["group_coord_y"] == ("shot_point", "trace")
