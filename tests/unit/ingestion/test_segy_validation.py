"""Tests for SEG-Y spec/template validation (canonical ingestion path)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from segy.schema import HeaderField
from segy.standards import get_segy_standard

from mdio.builder.template_registry import TemplateRegistry
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate
from mdio.ingestion.segy.validation import _validate_spec_in_template


class TestValidateSpecInTemplate:
    """Direct tests for the canonical ``mdio.ingestion.segy.validation`` module."""

    def test_passes_with_all_required_fields(self) -> None:
        """All declared dim/coord fields present → no error."""
        template = MagicMock(spec=AbstractDatasetTemplate)
        template.spatial_dimension_names = ("inline", "crossline")
        template.coordinate_names = ("cdp_x", "cdp_y")
        template.calculated_dimension_names = ()

        segy_spec = get_segy_standard(1.0)
        _validate_spec_in_template(segy_spec, template)

    def test_missing_fields_listed_in_error(self) -> None:
        """The error message must enumerate all missing required fields."""
        template = MagicMock(spec=AbstractDatasetTemplate)
        template.name = "CustomTemplate"
        template.spatial_dimension_names = ("custom_dim1", "custom_dim2")
        template.coordinate_names = ("custom_coord_x",)
        template.calculated_dimension_names = ()

        spec = get_segy_standard(1.0)
        # Only one of the custom dims is present
        spec = spec.customize(trace_header_fields=[HeaderField(name="custom_dim1", byte=189, format="int32")])

        with pytest.raises(ValueError, match=r"Required fields.*not found in.*segy_spec") as exc:
            _validate_spec_in_template(spec, template)

        msg = str(exc.value)
        assert "custom_dim2" in msg
        assert "custom_coord_x" in msg
        assert "CustomTemplate" in msg

    def test_missing_coordinate_scalar_raises(self) -> None:
        """A spec without ``coordinate_scalar`` must always fail."""
        template = MagicMock(spec=AbstractDatasetTemplate)
        template.name = "TestTemplate"
        template.spatial_dimension_names = ("inline", "crossline")
        template.coordinate_names = ("cdp_x", "cdp_y")
        template.calculated_dimension_names = ()

        spec = get_segy_standard(1.0)
        kept = [f for f in spec.trace.header.fields if f.name != "coordinate_scalar"]
        kept.append(HeaderField(name="not_coordinate_scalar", byte=71, format="int16"))
        spec = spec.customize(trace_header_fields=kept)

        with pytest.raises(ValueError, match=r"coordinate_scalar"):
            _validate_spec_in_template(spec, template)

    def test_calculated_dimensions_are_not_required(self) -> None:
        """Dimensions in ``calculated_dimension_names`` should not be required from the spec."""
        template = MagicMock(spec=AbstractDatasetTemplate)
        template.name = "CalcDim"
        template.spatial_dimension_names = ("inline", "crossline", "calculated_only")
        template.coordinate_names = ("cdp_x", "cdp_y")
        template.calculated_dimension_names = ("calculated_only",)

        segy_spec = get_segy_standard(1.0)
        _validate_spec_in_template(segy_spec, template)

    def test_obn_template_excludes_component_requirement(self) -> None:
        """OBN templates synthesize ``component`` when absent → not required from spec."""
        template = Seismic3DObnReceiverGathersTemplate(data_domain="time")
        # Make sure the registry has it (registry use is independent of validation).
        assert TemplateRegistry().get("ObnReceiverGathers3D") is not None

        spec = get_segy_standard(1.0)
        # Add all required OBN fields except 'component'.
        required = (set(template.spatial_dimension_names) | set(template.coordinate_names)) - set(
            template.calculated_dimension_names
        )
        required.discard("component")

        extra = [HeaderField(name=name, byte=189, format="int32") for name in sorted(required)]
        # Spread bytes so they don't collide.
        spec = spec.customize(
            trace_header_fields=[
                HeaderField(name=f.name, byte=189 + idx * 4, format="int32") for idx, f in enumerate(extra)
            ]
        )

        _validate_spec_in_template(spec, template)

    def test_obn_template_missing_other_required_field_still_fails(self) -> None:
        """Even with the ``component`` carve-out, other missing fields should error."""
        template = Seismic3DObnReceiverGathersTemplate(data_domain="time")
        spec = get_segy_standard(1.0)  # missing OBN-specific fields like 'receiver', 'shot_line', etc.

        with pytest.raises(ValueError, match=r"Required fields.*not found"):
            _validate_spec_in_template(spec, template)
