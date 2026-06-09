"""Unit tests for grid-override schema effects and their registry selection."""

from __future__ import annotations

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.templates.types import CoordinateSpec
from mdio.ingestion.schema import DimensionSpec
from mdio.ingestion.schema import ResolvedSchema
from mdio.ingestion.segy.index_strategies import IndexStrategyRegistry
from mdio.ingestion.segy.schema_effects import CollapseToTraceEffect
from mdio.ingestion.segy.schema_effects import InsertTraceDimEffect
from mdio.segy.geometry import GridOverrides


def _schema() -> ResolvedSchema:
    """A 3-spatial-dim + vertical schema with a coordinate spanning two spatial dims."""
    return ResolvedSchema(
        name="Toy",
        dimensions=[
            DimensionSpec(name="shot_point", is_spatial=True, dtype=ScalarType.UINT32),
            DimensionSpec(name="cable", is_spatial=True, dtype=ScalarType.UINT8),
            DimensionSpec(name="channel", is_spatial=True, dtype=ScalarType.UINT16),
            DimensionSpec(name="time", is_spatial=False, dtype=ScalarType.INT32),
        ],
        coordinates=[
            CoordinateSpec(
                name="group_coord_x",
                dimensions=("shot_point", "cable", "channel"),
                dtype=ScalarType.FLOAT64,
            ),
        ],
        chunk_shape=(8, 1, 128, 2048),
    )


class TestRegistrySchemaEffectSelection:
    """The registry maps GridOverrides to the matching SchemaEffect (single source)."""

    def test_no_overrides_returns_none(self) -> None:
        """A None or empty override yields no schema effect."""
        registry = IndexStrategyRegistry()
        assert registry.schema_effect(None) is None
        assert registry.schema_effect(GridOverrides()) is None

    def test_header_only_overrides_return_none(self) -> None:
        """Overrides that only transform headers (no layout change) yield no effect."""
        registry = IndexStrategyRegistry()
        assert registry.schema_effect(GridOverrides(auto_channel_wrap=True)) is None

    def test_non_binned_wires_chunksize_and_dims(self) -> None:
        """NonBinned yields a CollapseToTraceEffect carrying the override's chunksize and dims."""
        overrides = GridOverrides(non_binned=True, chunksize=128, non_binned_dims=["channel"])
        effect = IndexStrategyRegistry().schema_effect(overrides)
        assert isinstance(effect, CollapseToTraceEffect)
        assert effect.chunksize == 128
        assert effect.collapse_dims == ("channel",)

    def test_has_duplicates_inserts_trace(self) -> None:
        """HasDuplicates yields a 1-wide InsertTraceDimEffect."""
        effect = IndexStrategyRegistry().schema_effect(GridOverrides(has_duplicates=True))
        assert isinstance(effect, InsertTraceDimEffect)
        assert effect.chunksize == 1


class TestInsertTraceDimEffect:
    """InsertTraceDimEffect adds a calculated trace dim before the vertical axis."""

    def test_inserts_trace_before_vertical(self) -> None:
        """A 1-wide calculated trace dim is inserted; coordinates are untouched."""
        result = InsertTraceDimEffect(chunksize=1).apply(_schema())
        assert [d.name for d in result.dimensions] == ["shot_point", "cable", "channel", "trace", "time"]
        assert result.chunk_shape == (8, 1, 128, 1, 2048)
        trace = next(d for d in result.dimensions if d.name == "trace")
        assert trace.is_calculated is True
        # Coordinates are unchanged by duplicate handling.
        assert result.coordinates[0].dimensions == ("shot_point", "cable", "channel")


class TestCollapseToTraceEffect:
    """CollapseToTraceEffect collapses spatial dims into a single trace dim."""

    def test_default_collapses_all_but_first(self) -> None:
        """A None collapse set keeps the first spatial dim and collapses the rest."""
        result = CollapseToTraceEffect(chunksize=64, collapse_dims=None).apply(_schema())
        assert [d.name for d in result.dimensions] == ["shot_point", "trace", "time"]
        assert result.chunk_shape == (8, 64, 2048)

    def test_explicit_collapse_dims(self) -> None:
        """Only the named dims collapse; the coordinate is rewritten onto trace."""
        result = CollapseToTraceEffect(chunksize=128, collapse_dims=("channel",)).apply(_schema())
        assert [d.name for d in result.dimensions] == ["shot_point", "cable", "trace", "time"]
        group_coord_x = next(c for c in result.coordinates if c.name == "group_coord_x")
        assert group_coord_x.dimensions == ("shot_point", "cable", "trace")

    def test_collapsed_dim_becomes_trace_coordinate(self) -> None:
        """A collapsed dimension is re-expressed as a coordinate over trace with its dtype."""
        result = CollapseToTraceEffect(chunksize=64, collapse_dims=("channel",)).apply(_schema())
        channel = next(c for c in result.coordinates if c.name == "channel")
        assert channel.dimensions == ("trace",)
        assert channel.dtype == ScalarType.UINT16

    def test_empty_collapse_is_noop_for_dimensions(self) -> None:
        """An empty collapse set inserts no trace dim and keeps the original layout."""
        result = CollapseToTraceEffect(chunksize=64, collapse_dims=()).apply(_schema())
        assert [d.name for d in result.dimensions] == ["shot_point", "cable", "channel", "time"]
        assert result.chunk_shape == (8, 1, 128, 2048)
