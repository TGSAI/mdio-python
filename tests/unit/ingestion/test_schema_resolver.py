"""Unit tests for the v1.2 SchemaResolver."""

from __future__ import annotations

from mdio.builder.templates.seismic_3d_cdp import Seismic3DCdpGathersTemplate
from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate
from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate
from mdio.ingestion.schema import DimensionSpec
from mdio.ingestion.schema import ResolvedSchema
from mdio.ingestion.schema.resolver import SchemaResolver
from mdio.segy.geometry import GridOverrides


class TestSchemaResolverNoOverrides:
    """Resolving a template without grid overrides mirrors the template layout."""

    def test_streamer_shot_template_basic(self) -> None:
        """A plain template resolves to its dimensions, vertical axis, and chunk shape."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template, grid_overrides=None)

        assert schema.name == "StreamerShotGathers3D"
        assert [d.name for d in schema.dimensions] == ["shot_point", "cable", "channel", "time"]
        assert schema.dimensions[-1].is_spatial is False
        assert schema.dimensions[-1].is_calculated is False
        # Default chunk shape comes straight from the template.
        assert schema.chunk_shape == template.full_chunk_shape

    def test_obn_template_marks_shot_index_as_calculated(self) -> None:
        """The OBN template's ``shot_index`` resolves as a calculated spatial dimension."""
        template = Seismic3DObnReceiverGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template, grid_overrides=None)

        shot_index = next(d for d in schema.dimensions if d.name == "shot_index")
        assert shot_index.is_calculated is True
        assert shot_index.is_spatial is True

    def test_cdp_required_fields(self) -> None:
        """Required fields cover spatial dims and coordinates."""
        template = Seismic3DCdpGathersTemplate(data_domain="time", gather_domain="offset")
        schema = SchemaResolver().resolve(template, grid_overrides=None)

        # Spatial dim keys + coordinate keys.
        required = schema.required_fields()
        assert {"inline", "crossline", "offset", "cdp_x", "cdp_y"}.issubset(required)
        assert "coordinate_scalar" not in required


class TestSchemaResolverNonBinned:
    """NonBinned overrides collapse spatial dimensions into a single ``trace`` axis."""

    def test_default_collapse_keeps_first_spatial_dim(self) -> None:
        """Default NonBinned keeps the first spatial dim and collapses the rest into ``trace``."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        # Streamer shot template default chunk shape is (8, 1, 128, 2048).
        schema = SchemaResolver().resolve(template, GridOverrides(non_binned=True, chunksize=64))

        names = [d.name for d in schema.dimensions]
        assert names == ["shot_point", "trace", "time"]
        # shot_point keeps its original chunk (8); trace gets the override (64); vertical (2048) preserved.
        assert schema.chunk_shape == (8, 64, 2048)

    def test_explicit_non_binned_dims(self) -> None:
        """Explicit ``non_binned_dims`` collapse only the named dimensions into ``trace``."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        overrides = GridOverrides(non_binned=True, chunksize=128, non_binned_dims=["channel"])
        schema = SchemaResolver().resolve(template, overrides)

        names = [d.name for d in schema.dimensions]
        assert names == ["shot_point", "cable", "trace", "time"]
        # shot_point=8, cable=1 preserved; trace=128 (override); vertical=2048.
        assert schema.chunk_shape == (8, 1, 128, 2048)

    def test_coordinate_dimensions_collapsed_when_referenced(self) -> None:
        """Coordinates referencing collapsed dims are rewritten to depend on ``trace``."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template, GridOverrides(non_binned=True, chunksize=64))
        # group_coord_x originally depends on (shot_point, cable, channel). After NonBinned
        # collapses cable+channel, it should depend on (shot_point, trace).
        group_coord_x = next(c for c in schema.coordinates if c.name == "group_coord_x")
        assert group_coord_x.dimensions == ("shot_point", "trace")

    def test_grid_override_provenance_not_in_schema_metadata(self) -> None:
        """The resolver is mechanics-only: override provenance is attached at the dataset level."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        overrides = GridOverrides(non_binned=True, chunksize=64)
        schema = SchemaResolver().resolve(template, overrides)
        assert "gridOverrides" not in schema.metadata


class TestSchemaResolverHasDuplicates:
    """HasDuplicates overrides insert a 1-wide ``trace`` dimension before the vertical axis."""

    def test_inserts_trace_dim_with_chunksize_one(self) -> None:
        """HasDuplicates inserts a ``trace`` dim with chunksize 1 before the vertical dim."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template, GridOverrides(has_duplicates=True))

        names = [d.name for d in schema.dimensions]
        assert names == ["shot_point", "cable", "channel", "trace", "time"]
        # Streamer shot default chunks (8, 1, 128, 2048); trace dim is a 1-wide chunk inserted
        # before the vertical dim.
        assert schema.chunk_shape == (8, 1, 128, 1, 2048)

    def test_has_duplicates_provenance_not_in_schema_metadata(self) -> None:
        """The resolver does not record override provenance in schema metadata."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template, GridOverrides(has_duplicates=True))
        assert "gridOverrides" not in schema.metadata


class TestMissingCalculatedDimensions:
    """Tests for ``ResolvedSchema.missing_calculated_dimensions``."""

    def _schema(self) -> ResolvedSchema:
        return ResolvedSchema(
            name="Calc",
            dimensions=[
                DimensionSpec(name="receiver", is_spatial=True),
                DimensionSpec(name="shot_index", is_spatial=True, is_calculated=True),
                DimensionSpec(name="time", is_spatial=False),
            ],
            coordinates=[],
            chunk_shape=(2, 2, 4),
        )

    def test_reports_missing_calculated_dim(self) -> None:
        """A calculated dim absent from produced names is reported."""
        schema = self._schema()
        assert schema.missing_calculated_dimensions(["receiver", "time"]) == ["shot_index"]

    def test_empty_when_calculated_dim_produced(self) -> None:
        """Nothing is missing once the calculated dim is produced."""
        schema = self._schema()
        assert schema.missing_calculated_dimensions(["receiver", "shot_index", "time"]) == []

    def test_non_calculated_dims_are_never_reported(self) -> None:
        """A missing non-calculated spatial dim is not flagged by this check."""
        schema = self._schema()
        # 'receiver' is read from headers, not calculated, so its absence is not reported here.
        assert schema.missing_calculated_dimensions(["time"]) == ["shot_index"]
