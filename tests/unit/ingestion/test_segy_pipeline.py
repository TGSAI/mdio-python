"""Unit tests for the slim ingestion pipeline helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mdio.ingestion.schema import DimensionSpec
from mdio.ingestion.schema import ResolvedSchema
from mdio.ingestion.segy import pipeline
from mdio.ingestion.segy.raw_headers import build_raw_header_variables


def _schema(dimensions: list[DimensionSpec]) -> ResolvedSchema:
    return ResolvedSchema(
        name="Toy",
        dimensions=dimensions,
        coordinates=[],
        chunk_shape=tuple(2 for _ in dimensions),
    )


class TestResolveOutputPath:
    """Tests for ``_resolve_output_path`` overwrite enforcement."""

    def test_returns_path_when_absent(self, tmp_path) -> None:  # noqa: ANN001
        """A non-existent location is returned normalized."""
        target = tmp_path / "out.mdio"
        result = pipeline._resolve_output_path(str(target), overwrite=False)
        assert result.as_posix().endswith("out.mdio")

    def test_raises_when_exists_without_overwrite(self, tmp_path) -> None:  # noqa: ANN001
        """An existing location without overwrite raises FileExistsError."""
        target = tmp_path / "out.mdio"
        target.mkdir()
        with pytest.raises(FileExistsError, match="overwrite=True"):
            pipeline._resolve_output_path(str(target), overwrite=False)

    def test_allows_existing_with_overwrite(self, tmp_path) -> None:  # noqa: ANN001
        """An existing location with overwrite is allowed."""
        target = tmp_path / "out.mdio"
        target.mkdir()
        result = pipeline._resolve_output_path(str(target), overwrite=True)
        assert result.as_posix().endswith("out.mdio")


class TestVerifyCalculatedDimensions:
    """Tests for ``_verify_calculated_dimensions``."""

    def test_passes_when_calculated_dim_present(self) -> None:
        """No error when every calculated dim was produced."""
        schema = _schema(
            [
                DimensionSpec(name="receiver", is_spatial=True),
                DimensionSpec(name="shot_index", is_spatial=True, is_calculated=True),
                DimensionSpec(name="time", is_spatial=False),
            ]
        )
        produced = [SimpleNamespace(name=n) for n in ("receiver", "shot_index", "time")]
        pipeline._verify_calculated_dimensions(schema, produced, "Toy")

    def test_raises_when_calculated_dim_missing(self) -> None:
        """A missing calculated dim raises a descriptive ValueError."""
        schema = _schema(
            [
                DimensionSpec(name="receiver", is_spatial=True),
                DimensionSpec(name="shot_index", is_spatial=True, is_calculated=True),
                DimensionSpec(name="time", is_spatial=False),
            ]
        )
        produced = [SimpleNamespace(name=n) for n in ("receiver", "time")]
        with pytest.raises(ValueError, match="shot_index"):
            pipeline._verify_calculated_dimensions(schema, produced, "Toy")


class TestBuildRawHeaderVariables:
    """Tests for the isolated experimental raw-headers feature."""

    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the feature flag off, no extra variables are produced."""
        monkeypatch.delenv("MDIO__IMPORT__RAW_HEADERS", raising=False)
        schema = _schema(
            [
                DimensionSpec(name="inline", is_spatial=True),
                DimensionSpec(name="crossline", is_spatial=True),
                DimensionSpec(name="time", is_spatial=False),
            ]
        )
        assert build_raw_header_variables(schema) == []
