"""Tests for SEG-Y export revision-field handling in ``mdio.segy.creation``."""

from __future__ import annotations

import pytest
from segy.factory import SegyFactory
from segy.standards import get_segy_standard

from mdio.segy.creation import _prepare_export_revision

_MDIO_REV = {"segy_revision_major": 1, "segy_revision_minor": 0}


class TestPrepareExportRevision:
    """Revision encoding must match the spec's own field names, not always rev 1."""

    @pytest.mark.parametrize("revision", [2, 2.1])
    def test_rev2_specs_keep_major_minor(self, revision: float) -> None:
        """Rev 2 / 2.1 keep ``segy_revision_major`` / ``minor``; factory can write header."""
        spec = get_segy_standard(revision)
        result = _prepare_export_revision(
            spec,
            {"sample_interval": 2000, "samples_per_trace": 1, "segy_revision_major": 2, "segy_revision_minor": 0},
        )

        names = spec.binary_header.names
        assert "segy_revision_major" in names
        assert "segy_revision_minor" in names
        assert "segy_revision" not in names
        assert "segy_revision" not in result

        factory = SegyFactory(spec=spec, sample_interval=2000, samples_per_trace=1)
        assert len(factory.create_binary_header(result)) == 400

    def test_rev1_spec_encodes_combined_field(self) -> None:
        """Rev 1 spec encodes MDIO major/minor into combined ``segy_revision``."""
        spec = get_segy_standard(1)
        result = _prepare_export_revision(spec, dict(_MDIO_REV))

        assert result["segy_revision"] == (1 << 8)
        assert "segy_revision_major" not in result

    def test_spec_without_revision_fields_gets_rev1_field(self) -> None:
        """Custom spec with no revision fields gets rev-1 ``segy_revision`` injected."""
        spec = get_segy_standard(1)
        spec.binary_header.remove_field("segy_revision")
        result = _prepare_export_revision(spec, dict(_MDIO_REV))

        assert "segy_revision" in spec.binary_header.names
        assert result["segy_revision"] == (1 << 8)
