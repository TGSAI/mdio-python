"""Unit tests for concrete seismic dataset template implementations."""

import pytest

from mdio.builder.template_registry import TemplateRegistry
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.seismic_2d_poststack import Seismic2DPostStackTemplate


class TestSeismicTemplates:
    """Test cases for Seismic2DPostStackTemplate."""

    def test_chunk_shape_assignment(self) -> None:
        """Test that chunk shape is assigned correctly."""
        template = Seismic2DPostStackTemplate("time")
        template.build_dataset("test", (50, 50))
        template.full_chunk_shape = (32, 32)

        assert template._var_chunk_shape == (32, 32)

    def test_chunk_shape_assignment_exception(self) -> None:
        """Test that chunk shape assignment raises exception for invalid dimensions."""
        template = Seismic2DPostStackTemplate("time")
        template.build_dataset("test", (50, 50))

        with pytest.raises(ValueError, match="Chunk shape.*has.*dimensions, expected"):
            template.full_chunk_shape = (32, 32, 32)

    def test_chunk_shape_with_minus_one_before_build(self) -> None:
        """Test that chunk shape can be set with -1 before build_dataset."""
        template = Seismic2DPostStackTemplate("time")

        # Should be able to set chunk shape with -1 before build_dataset
        template.full_chunk_shape = (32, -1)

        # Before build_dataset, getter should return unexpanded values
        assert template.full_chunk_shape == (32, -1)
        assert template._var_chunk_shape == (32, -1)

    def test_chunk_shape_with_minus_one_after_build(self) -> None:
        """Test that -1 values are expanded after build_dataset."""
        template = Seismic2DPostStackTemplate("time")
        template.full_chunk_shape = (32, -1)

        # Build dataset with specific sizes
        template.build_dataset("test", (100, 200))

        # After build_dataset, getter should expand -1 to dimension size
        assert template.full_chunk_shape == (32, 200)
        assert template._var_chunk_shape == (32, -1)  # Internal storage unchanged

    def test_chunk_shape_validation_invalid_values(self) -> None:
        """Test that chunk shape setter rejects invalid values."""
        template = Seismic2DPostStackTemplate("time")
        template.build_dataset("test", (50, 50))

        # Test rejection of 0
        with pytest.raises(ValueError, match="Chunk size must be positive integer or -1"):
            template.full_chunk_shape = (32, 0)

        # Test rejection of negative values other than -1
        with pytest.raises(ValueError, match="Chunk size must be positive integer or -1"):
            template.full_chunk_shape = (32, -2)

        # Test that positive values and -1 are accepted
        template.full_chunk_shape = (32, -1)  # Should not raise
        template.full_chunk_shape = (32, 16)  # Should not raise

    def test_all_templates_inherit_from_abstract(self) -> None:
        """Test that all concrete templates inherit from AbstractDatasetTemplate."""
        registry = TemplateRegistry()
        template_names = registry.list_all_templates()

        for template_name in template_names:
            template = registry.get(template_name)
            assert isinstance(template, AbstractDatasetTemplate)
            # That each template has the required properties and methods
            assert hasattr(template, "name")
            assert hasattr(template, "default_variable_name")
            assert hasattr(template, "trace_domain")
            assert hasattr(template, "dimension_names")
            assert hasattr(template, "coordinate_names")
            assert hasattr(template, "build_dataset")

        assert len(template_names) == len(set(template_names)), f"Duplicate template names found: {template_names}"
