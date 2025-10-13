"""Tests for SEG-Y spec validation against MDIO templates."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from segy.schema import HeaderField
from segy.standards import get_segy_standard

from mdio.converters.segy import _validate_spec_in_template


class TestValidateSpecInTemplate:
    """Test cases for _validate_spec_in_template function."""

    def test_validation_passes_with_all_required_fields(self) -> None:
        """Test that validation passes when all required fields are present."""
        template = MagicMock()
        template._dim_names = ("inline", "crossline", "time")
        template._coord_names = ("cdp_x", "cdp_y")

        # SegySpec with all required fields
        spec = get_segy_standard(1.0)
        header_fields = [
            HeaderField(name="inline", byte=189, format="int32"),
            HeaderField(name="crossline", byte=193, format="int32"),
            HeaderField(name="cdp_x", byte=181, format="int32"),
            HeaderField(name="cdp_y", byte=185, format="int32"),
        ]
        segy_spec = spec.customize(trace_header_fields=header_fields)

        # Should not raise any exception
        _validate_spec_in_template(segy_spec, template)

    def test_validation_fails_with_missing_fields(self) -> None:
        """Test that validation fails when required fields are missing."""
        # Template requiring custom fields not in standard spec
        template = MagicMock()
        template.name = "CustomTemplate"
        template._dim_names = ("custom_dim1", "custom_dim2", "time")
        template._coord_names = ("custom_coord_x", "custom_coord_y")

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
