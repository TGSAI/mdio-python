"""Tests for the schema resolution system."""

import pytest

from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate
from mdio.ingestion.schema_resolver import SchemaResolver
from mdio.segy.geometry import GridOverrides


class TestSchemaResolver:
    """Tests for SchemaResolver class."""

    def test_resolve_without_overrides(self) -> None:
        """Test resolving schema from template without grid overrides."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        schema = resolver.resolve(template, grid_overrides=None)

        assert schema.name == "StreamerShotGathers3D"
        assert len(schema.dimensions) == 4
        assert schema.dimensions[0].name == "shot_point"
        assert schema.dimensions[1].name == "cable"
        assert schema.dimensions[2].name == "channel"
        assert schema.dimensions[3].name == "time"
        assert schema.dimensions[3].is_spatial is False
        assert schema.chunk_shape == (8, 1, 128, 2048)

    def test_resolve_with_non_binned_default(self) -> None:
        """Test resolving schema with non_binned override (default behavior)."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        overrides = GridOverrides(non_binned=True, chunksize=64)
        schema = resolver.resolve(template, grid_overrides=overrides)

        # Should replace cable and channel with trace (default: all but first)
        assert len(schema.dimensions) == 3
        assert schema.dimensions[0].name == "shot_point"
        assert schema.dimensions[1].name == "trace"
        assert schema.dimensions[1].source == "computed"
        assert schema.dimensions[2].name == "time"
        assert schema.chunk_shape == (8, 64, 2048)

    def test_resolve_with_non_binned_custom_replace_dims(self) -> None:
        """Test resolving schema with non_binned override and custom replace_dims."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        overrides = GridOverrides(non_binned=True, chunksize=128, replace_dims=["channel"])
        schema = resolver.resolve(template, grid_overrides=overrides)

        # Should replace only channel with trace
        assert len(schema.dimensions) == 4
        assert schema.dimensions[0].name == "shot_point"
        assert schema.dimensions[1].name == "cable"
        assert schema.dimensions[2].name == "trace"
        assert schema.dimensions[3].name == "time"
        assert schema.chunk_shape == (8, 1, 128, 2048)

    def test_resolve_with_has_duplicates(self) -> None:
        """Test resolving schema with has_duplicates override."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        overrides = GridOverrides(has_duplicates=True)
        schema = resolver.resolve(template, grid_overrides=overrides)

        # Should add trace dimension with chunksize=1
        assert len(schema.dimensions) == 5
        assert schema.dimensions[0].name == "shot_point"
        assert schema.dimensions[1].name == "cable"
        assert schema.dimensions[2].name == "channel"
        assert schema.dimensions[3].name == "trace"
        assert schema.dimensions[3].source == "computed"
        assert schema.dimensions[4].name == "time"
        assert schema.chunk_shape == (8, 1, 128, 1, 2048)

    def test_required_header_fields(self) -> None:
        """Test that required_header_fields returns correct fields."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        schema = resolver.resolve(template, grid_overrides=None)
        required = schema.required_header_fields()

        # Should include dimension headers and coordinate headers
        assert "shot_point" in required
        assert "cable" in required
        assert "channel" in required
        assert "gun" in required
        assert "source_coord_x" in required
        assert "source_coord_y" in required
        assert "group_coord_x" in required
        assert "group_coord_y" in required
        assert "coordinate_scalar" in required  # Always required

    def test_spatial_dimensions(self) -> None:
        """Test that spatial_dimensions returns only spatial dims."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        schema = resolver.resolve(template, grid_overrides=None)
        spatial = schema.spatial_dimensions()

        assert len(spatial) == 3
        assert all(d.is_spatial for d in spatial)
        assert spatial[0].name == "shot_point"
        assert spatial[1].name == "cable"
        assert spatial[2].name == "channel"

    def test_computed_dimensions(self) -> None:
        """Test that computed_dimensions returns only computed dims."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        # Without overrides, no computed dimensions
        schema = resolver.resolve(template, grid_overrides=None)
        computed = schema.computed_dimensions()
        assert len(computed) == 0

        # With non_binned, trace is computed
        overrides = GridOverrides(non_binned=True, chunksize=64)
        schema = resolver.resolve(template, grid_overrides=overrides)
        computed = schema.computed_dimensions()
        assert len(computed) == 1
        assert computed[0].name == "trace"

    def test_metadata_includes_grid_overrides(self) -> None:
        """Test that grid overrides are included in metadata."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        overrides = GridOverrides(non_binned=True, chunksize=64)
        schema = resolver.resolve(template, grid_overrides=overrides)

        assert "gridOverrides" in schema.metadata
        assert schema.metadata["gridOverrides"]["NonBinned"] is True
        assert schema.metadata["gridOverrides"]["chunksize"] == 64


class TestNonBinnedCoordinateDimensionUpdates:
    """Test that NonBinned properly updates coordinate dimensions."""

    def test_coordinate_dimensions_updated_after_collapse(self) -> None:
        """Test coordinates referencing collapsed dims get trace dimension."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        # Default NonBinned collapses cable and channel
        overrides = GridOverrides(non_binned=True, chunksize=64)
        schema = resolver.resolve(template, grid_overrides=overrides)

        # Coordinates that didn't reference collapsed dims should stay the same
        gun = next(c for c in schema.coordinates if c.name == "gun")
        assert gun.dimensions == ("shot_point",)

        source_x = next(c for c in schema.coordinates if c.name == "source_coord_x")
        assert source_x.dimensions == ("shot_point",)

        # Coordinates that referenced collapsed dims should have trace appended
        group_x = next(c for c in schema.coordinates if c.name == "group_coord_x")
        assert group_x.dimensions == ("shot_point", "trace")

        group_y = next(c for c in schema.coordinates if c.name == "group_coord_y")
        assert group_y.dimensions == ("shot_point", "trace")

    def test_coordinate_dimensions_with_custom_replace_dims(self) -> None:
        """Test coordinate dimension updates with custom replace_dims."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        # Only collapse channel
        overrides = GridOverrides(non_binned=True, chunksize=128, replace_dims=["channel"])
        schema = resolver.resolve(template, grid_overrides=overrides)

        # group_coord_x originally had (shot_point, cable, channel)
        # After collapsing channel, should have (shot_point, cable, trace)
        group_x = next(c for c in schema.coordinates if c.name == "group_coord_x")
        assert group_x.dimensions == ("shot_point", "cable", "trace")

    def test_coordinate_dimensions_when_all_collapsed(self) -> None:
        """Test coordinate dimensions when all spatial dims are collapsed."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        # Collapse all spatial dimensions
        overrides = GridOverrides(non_binned=True, chunksize=256, replace_dims=["shot_point", "cable", "channel"])
        schema = resolver.resolve(template, grid_overrides=overrides)

        # All coordinates should now only reference trace
        for coord in schema.coordinates:
            if coord.name in ("group_coord_x", "group_coord_y"):
                # These originally had all three spatial dims
                assert coord.dimensions == ("trace",)
            elif coord.name in ("gun", "source_coord_x", "source_coord_y"):
                # These originally had just shot_point
                assert coord.dimensions == ("trace",)


class TestCoordinateDimensionInference:
    """Test coordinate dimension inference logic."""

    def test_cdp_coordinates_2d(self) -> None:
        """Test CDP coordinate dimensions for 2D templates."""
        from mdio.builder.templates.seismic_2d_cdp import Seismic2DCdpGathersTemplate

        template = Seismic2DCdpGathersTemplate(data_domain="time", gather_domain="offset")
        resolver = SchemaResolver()

        schema = resolver.resolve(template, grid_overrides=None)

        # Find cdp_x coordinate
        cdp_x = next(c for c in schema.coordinates if c.name == "cdp_x")
        assert cdp_x.dimensions == ("cdp",)

    def test_cdp_coordinates_3d(self) -> None:
        """Test CDP coordinate dimensions for 3D templates."""
        from mdio.builder.templates.seismic_3d_cdp import Seismic3DCdpGathersTemplate

        template = Seismic3DCdpGathersTemplate(data_domain="time", gather_domain="offset")
        resolver = SchemaResolver()

        schema = resolver.resolve(template, grid_overrides=None)

        # Find cdp_x coordinate
        cdp_x = next(c for c in schema.coordinates if c.name == "cdp_x")
        assert cdp_x.dimensions == ("inline", "crossline")

    def test_source_coordinates(self) -> None:
        """Test source coordinate dimensions."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        schema = resolver.resolve(template, grid_overrides=None)

        source_x = next(c for c in schema.coordinates if c.name == "source_coord_x")
        assert source_x.dimensions == ("shot_point",)

    def test_group_coordinates(self) -> None:
        """Test group coordinate dimensions."""
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()

        schema = resolver.resolve(template, grid_overrides=None)

        group_x = next(c for c in schema.coordinates if c.name == "group_coord_x")
        assert group_x.dimensions == ("shot_point", "cable", "channel")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
