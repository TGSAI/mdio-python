"""Guardrail: ingestion must never mutate template internals.

The pre-v1.2 ingestion path mutated templates in place (reassigning ``_dim_names``,
``_var_chunk_shape``, ``_logical_coord_names``) and even monkeypatched template methods at
runtime. The refactor replaced that with an immutable ``ResolvedSchema``. This test fails
loudly if any ingestion module reintroduces template-internals mutation or method rebinding.
"""

from __future__ import annotations

import re
from pathlib import Path

import mdio

# Assignment to a private template attribute, e.g. ``template._dim_names = ...`` or
# rebinding a method/attribute on a template object, e.g. ``template._add_coordinates = ...``.
_MUTATION = re.compile(r"\b(?:template|mdio_template)\.\w+\s*=(?!=)")

_INGESTION_ROOT = Path(mdio.__file__).parent / "ingestion"


def _ingestion_sources() -> list[Path]:
    return sorted(p for p in _INGESTION_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_ingestion_never_mutates_template() -> None:
    """No ingestion module assigns to a template attribute (mutation/monkeypatch guard)."""
    offenders: list[str] = []
    for path in _ingestion_sources():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            if _MUTATION.search(code):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")

    assert not offenders, "Ingestion must not mutate template internals:\n" + "\n".join(offenders)
