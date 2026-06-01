"""Unit tests for the v1.2 SchemaResolver."""

from __future__ import annotations

from mdio.builder.templates.seismic_3d_cdp import Seismic3DCdpGathersTemplate
from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate
from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate
from mdio.ingestion.schema_resolver import SchemaResolver
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

    def test_cdp_required_header_fields(self) -> None:
        """Required header fields cover spatial dims, coordinates, and ``coordinate_scalar``."""
        template = Seismic3DCdpGathersTemplate(data_domain="time", gather_domain="offset")
        schema = SchemaResolver().resolve(template, grid_overrides=None)

        # Spatial dim header keys + coordinate header keys + always-present coordinate_scalar.
        required = schema.required_header_fields()
        assert {"inline", "crossline", "offset", "cdp_x", "cdp_y", "coordinate_scalar"}.issubset(required)


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

    def test_non_binned_flag_recorded_in_metadata(self) -> None:
        """The NonBinned flag is recorded under ``gridOverrides`` metadata."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        overrides = GridOverrides(non_binned=True, chunksize=64)
        schema = SchemaResolver().resolve(template, overrides)
        assert "gridOverrides" in schema.metadata
        assert schema.metadata["gridOverrides"].get("NonBinned") is True


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

    def test_has_duplicates_metadata(self) -> None:
        """The HasDuplicates flag is recorded under ``gridOverrides`` metadata."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template, GridOverrides(has_duplicates=True))
        assert schema.metadata["gridOverrides"].get("HasDuplicates") is True
