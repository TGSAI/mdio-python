"""SegySpec/template validation for SEG-Y ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdio.segy.scalar import SCALE_COORDINATE_KEYS

if TYPE_CHECKING:
    from segy.schema import SegySpec

    from mdio.builder.templates.base import AbstractDatasetTemplate


def validate_spec_in_template(segy_spec: SegySpec, mdio_template: AbstractDatasetTemplate) -> None:
    """Validate that the SegySpec has all required fields in the MDIO template."""
    # Import here to avoid circular imports at module load time
    from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate  # noqa: PLC0415

    header_fields = {field.name for field in segy_spec.trace.header.fields}

    required_fields = set(mdio_template.spatial_dimension_names) | set(mdio_template.coordinate_names)
    required_fields = required_fields - set(mdio_template.calculated_dimension_names)

    # 'component' is optional for OBN (synthesized if missing)
    if isinstance(mdio_template, Seismic3DObnReceiverGathersTemplate):
        required_fields.discard("component")

    if any(field in SCALE_COORDINATE_KEYS for field in required_fields):
        required_fields = required_fields | {"coordinate_scalar"}
    missing_fields = required_fields - header_fields

    if missing_fields:
        err = (
            f"Required fields {sorted(missing_fields)} for template {mdio_template.name} "
            f"not found in the provided segy_spec"
        )
        raise ValueError(err)
