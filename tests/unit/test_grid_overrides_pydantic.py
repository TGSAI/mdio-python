"""Unit tests for the typed :class:`mdio.GridOverrides` Pydantic model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from mdio.converters.segy import _coerce_grid_overrides
from mdio.segy.geometry import GridOverrides

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


def test_grid_overrides_defaults() -> None:
    """Default instance has every flag off and is falsy."""
    overrides = GridOverrides()
    assert not overrides.auto_channel_wrap
    assert not overrides.auto_shot_wrap
    assert not overrides.calculate_shot_index
    assert not overrides.non_binned
    assert not overrides.has_duplicates
    assert overrides.chunksize is None
    assert overrides.non_binned_dims is None
    assert not bool(overrides)


def test_grid_overrides_aliases() -> None:
    """Legacy CamelCase aliases populate the snake_case fields."""
    overrides = GridOverrides(AutoChannelWrap=True, chunksize=64)
    assert overrides.auto_channel_wrap is True
    assert overrides.chunksize == 64
    assert bool(overrides) is True


def test_grid_overrides_calculate_shot_index_alias() -> None:
    """OBN-specific ``CalculateShotIndex`` survives the typed shape."""
    overrides = GridOverrides(CalculateShotIndex=True)
    assert overrides.calculate_shot_index is True
    assert bool(overrides) is True


def test_grid_overrides_validation() -> None:
    """``chunksize`` must be strictly positive."""
    with pytest.raises(ValidationError):
        GridOverrides(chunksize=0)

    with pytest.raises(ValidationError):
        GridOverrides(chunksize=-1)


def test_grid_overrides_rejects_unknown_keys() -> None:
    """Unknown keys are rejected at construction by ``extra='forbid'``."""
    with pytest.raises(ValidationError):
        GridOverrides.model_validate({"FutureFlag": True})


def test_grid_overrides_serialization() -> None:
    """``model_dump`` round-trips both legacy and modern key shapes."""
    overrides = GridOverrides(AutoChannelWrap=True, chunksize=64)

    dumped_legacy = overrides.model_dump(by_alias=True, exclude_defaults=True)
    assert dumped_legacy == {"AutoChannelWrap": True, "chunksize": 64}

    dumped_modern = overrides.model_dump(exclude_defaults=True)
    assert dumped_modern == {"auto_channel_wrap": True, "chunksize": 64}


def test_grid_overrides_to_legacy_dict() -> None:
    """``to_legacy_dict`` produces the dict shape consumed by ``GridOverrider``."""
    overrides = GridOverrides(non_binned=True, chunksize=128, non_binned_dims=["offset", "azimuth"])
    assert overrides.to_legacy_dict() == {
        "NonBinned": True,
        "chunksize": 128,
        "non_binned_dims": ["offset", "azimuth"],
    }


def test_grid_overrides_to_legacy_dict_default_is_empty() -> None:
    """Default instance dumps to an empty dict."""
    assert GridOverrides().to_legacy_dict() == {}


def test_grid_overrides_legacy_dict_roundtrip() -> None:
    """A legacy dict survives ``model_validate``/``to_legacy_dict`` unchanged."""
    legacy = {
        "CalculateShotIndex": True,
        "NonBinned": True,
        "chunksize": 64,
        "non_binned_dims": ["offset"],
    }
    assert GridOverrides.model_validate(legacy).to_legacy_dict() == legacy


def test_coerce_grid_overrides_converts_dict_with_log(caplog: LogCaptureFixture) -> None:
    """A dict input is coerced to :class:`GridOverrides` and a deprecation is logged."""
    legacy = {"CalculateShotIndex": True}
    with caplog.at_level(logging.WARNING, logger="mdio.converters.segy"):
        result = _coerce_grid_overrides(legacy)
    assert isinstance(result, GridOverrides)
    assert result.calculate_shot_index is True
    assert any("deprecated" in record.message for record in caplog.records)


def test_coerce_grid_overrides_rejects_unknown_dict_keys() -> None:
    """Dict inputs with unknown keys fail loudly instead of silently dropping them."""
    with pytest.raises(ValidationError):
        _coerce_grid_overrides({"FutureFlag": True})


def test_coerce_grid_overrides_passes_pydantic_model_through() -> None:
    """A :class:`GridOverrides` instance is returned unchanged."""
    overrides = GridOverrides(auto_channel_wrap=True)
    assert _coerce_grid_overrides(overrides) is overrides


def test_coerce_grid_overrides_none_returns_none() -> None:
    """``None`` round-trips to ``None``."""
    assert _coerce_grid_overrides(None) is None
