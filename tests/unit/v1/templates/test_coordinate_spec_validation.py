"""Tests for the temporary declare_coordinate_specs / _add_coordinates drift guard.

These tests pin the safety net that keeps ``declare_coordinate_specs`` in sync with the
coordinates actually built by ``_add_coordinates``, including for user-defined templates.
The guard (and these tests) are removed once the ingestion pipeline builds datasets directly
from the resolved schema.
"""

from __future__ import annotations

from typing import Any

import pytest

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import CoordinateSpec


class _SubsetCoordTemplate(AbstractDatasetTemplate):
    """User-style template whose coordinate spans a subset of the spatial dimensions."""

    def __init__(self, *, declare_correct_specs: bool) -> None:
        super().__init__(data_domain="time")
        self._declare_correct_specs = declare_correct_specs
        self._dim_names = ("shot", "channel", "time")
        self._physical_coord_names = ("src_x",)
        self._var_chunk_shape = (8, 8, 128)

    @property
    def _name(self) -> str:
        return "SubsetCoordTest"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {}

    def declare_coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        """Declare ``src_x`` over the correct subset, or fall back to the (wrong) default."""
        if self._declare_correct_specs:
            return (CoordinateSpec(name="src_x", dimensions=("shot",), dtype=ScalarType.FLOAT64),)
        return super().declare_coordinate_specs()

    def _add_coordinates(self) -> None:
        for name in self._dim_names:
            self._builder.add_coordinate(name, dimensions=(name,), data_type=ScalarType.INT32)
        # src_x is indexed by `shot` only, a subset of the spatial dims (shot, channel).
        self._builder.add_coordinate("src_x", dimensions=("shot",), data_type=ScalarType.FLOAT64)


def test_matching_specs_build_successfully() -> None:
    """A template whose declared specs match its built coordinates builds without error."""
    template = _SubsetCoordTemplate(declare_correct_specs=True)
    dataset = template.build_dataset("ok", sizes=(4, 8, 128))
    assert any(v.name == "src_x" for v in dataset.variables)


def test_dimension_drift_is_rejected() -> None:
    """A template relying on the default specs while building a subset-indexed coordinate fails."""
    template = _SubsetCoordTemplate(declare_correct_specs=False)
    with pytest.raises(ValueError, match="declares coordinate 'src_x' over dimensions"):
        template.build_dataset("drift", sizes=(4, 8, 128))


def test_missing_declaration_is_rejected() -> None:
    """A coordinate built but never declared is reported as out of sync."""

    class _UndeclaredCoordTemplate(_SubsetCoordTemplate):
        def declare_coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
            return ()

    template = _UndeclaredCoordTemplate(declare_correct_specs=True)
    with pytest.raises(ValueError, match="out of sync"):
        template.build_dataset("missing", sizes=(4, 8, 128))
