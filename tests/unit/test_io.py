"""Tests for low-level MDIO API I/O helpers."""

from __future__ import annotations

from types import MappingProxyType

from upath import UPath

from mdio.api.io import _normalize_storage_options


def test_normalize_storage_options_is_not_mappingproxy() -> None:
    """Storage options must not be a mappingproxy.

    `UPath.storage_options` returns a read-only ``mappingproxy`` that cannot be pickled. Blocked-I/O
    ingestion passes these options into ``ProcessPoolExecutor`` initargs, so a mappingproxy breaks
    spawned workers with ``TypeError: cannot pickle 'mappingproxy' object``.
    """
    storage_options = _normalize_storage_options(UPath("s3://bucket/key", key="access", secret="secret"))  # noqa: S106

    assert not isinstance(storage_options, MappingProxyType)
