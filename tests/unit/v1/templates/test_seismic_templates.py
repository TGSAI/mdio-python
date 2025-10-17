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

        with pytest.raises(ValueError, match="Chunk shape.*does not match dimension sizes"):
            template.full_chunk_shape = (32, 32, 32)

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
