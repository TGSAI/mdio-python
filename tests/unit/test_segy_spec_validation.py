"""Tests for SEG-Y spec validation against MDIO templates."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from segy.schema import HeaderField
from segy.standards import get_segy_standard

from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.converters.segy import _validate_spec_in_template


class TestValidateSpecInTemplate:
    """Test cases for _validate_spec_in_template function."""

    def test_validation_passes_with_all_required_fields(self) -> None:
        """Test that validation passes when all required fields are present."""
        template = MagicMock(spec=AbstractDatasetTemplate)
        template.spatial_dimension_names = ("inline", "crossline")
        template.coordinate_names = ("cdp_x", "cdp_y")

        # Use base SEG-Y standard which includes coordinate_scalar at byte 71
        segy_spec = get_segy_standard(1.0)

        # Should not raise any exception
        _validate_spec_in_template(segy_spec, template)

    def test_validation_fails_with_missing_fields(self) -> None:
        """Test that validation fails when required fields are missing."""
        # Template requiring custom fields not in standard spec
        template = MagicMock(spec=AbstractDatasetTemplate)
        template.name = "CustomTemplate"
        template.spatial_dimension_names = ("custom_dim1", "custom_dim2")
        template.coordinate_names = ("custom_coord_x", "custom_coord_y")

        # SegySpec with only one of the required custom fields
        spec = get_segy_standard(1.0)
        header_fields = [
            HeaderField(name="custom_dim1", byte=189, format="int32"),
        ]
        segy_spec = spec.customize(trace_header_fields=header_fields)

        # Should raise ValueError listing the missing fields
        with pytest.raises(ValueError, match=r"Required fields.*not found in.*segy_spec") as exc_info:
            _validate_spec_in_template(segy_spec, template)

        error_message = str(exc_info.value)
        assert "custom_dim2" in error_message
        assert "custom_coord_x" in error_message
        assert "custom_coord_y" in error_message
        assert "CustomTemplate" in error_message

    def test_validation_fails_with_missing_coordinate_scalar(self) -> None:
        """Test that validation fails when coordinate_scalar is missing, even with all other fields."""
        template = MagicMock(spec=AbstractDatasetTemplate)
        template.name = "TestTemplate"
        template.spatial_dimension_names = ("inline", "crossline")
        template.coordinate_names = ("cdp_x", "cdp_y")

        # Create SegySpec with all standard fields except coordinate_scalar
        spec = get_segy_standard(1.0)
        # Remove coordinate_scalar from the standard fields
        standard_fields = [field for field in spec.trace.header.fields if field.name != "coordinate_scalar"]
        standard_fields.append(HeaderField(name="not_coordinate_scalar", byte=71, format="int16"))
        segy_spec = spec.customize(trace_header_fields=standard_fields)

        # Should raise ValueError for missing coordinate_scalar
        with pytest.raises(ValueError, match=r"Required fields.*not found in.*segy_spec") as exc_info:
            _validate_spec_in_template(segy_spec, template)

        error_message = str(exc_info.value)
        assert "coordinate_scalar" in error_message
        assert "TestTemplate" in error_message
