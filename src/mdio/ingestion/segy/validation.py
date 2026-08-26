"""SegySpec/template validation for SEG-Y ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdio.segy.scalar import SCALE_COORDINATE_KEYS

if TYPE_CHECKING:
    from segy.schema import SegySpec

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.ingestion.schema.models import ResolvedSchema


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

    # Optional coordinates (e.g. 'gun' on streamer shot gathers) are populated only when the
    # source carries them; their absence must not fail ingestion.
    required_fields -= set(mdio_template.optional_coordinate_names)

    if any(field in SCALE_COORDINATE_KEYS for field in required_fields):
        required_fields = required_fields | {"coordinate_scalar"}
    missing_fields = required_fields - header_fields

    if missing_fields:
        err = (
            f"Required fields {sorted(missing_fields)} for template {mdio_template.name} "
            f"not found in the provided segy_spec"
        )
        raise ValueError(err)


def prune_absent_optional_coordinates(
    schema: ResolvedSchema, segy_spec: SegySpec, mdio_template: AbstractDatasetTemplate
) -> ResolvedSchema:
    """Drop optional coordinates the SEG-Y doesn't carry, so they aren't built as empty vars.

    An optional coordinate (see :attr:`AbstractDatasetTemplate.optional_coordinate_names`) is
    declared on the template so it can be populated *when present*, but if the source lacks the
    header field it would otherwise be materialized as an all-fill coordinate. Pruning it here
    keeps the built dataset honest: the coordinate exists iff the data actually supplied it.
    Returns ``schema`` unchanged when there is nothing to prune.
    """
    optional = set(mdio_template.optional_coordinate_names)
    if not optional:
        return schema
    header_fields = {field.name for field in segy_spec.trace.header.fields}
    kept = [c for c in schema.coordinates if c.name not in optional or c.name in header_fields]
    if len(kept) == len(schema.coordinates):
        return schema
    return schema.model_copy(update={"coordinates": kept})
