"""Deterministic planning for multi-shard SEG-Y -> single MDIO consolidation.

ADDITIVE and format-agnostic. Given each shard's *global* index coverage and the store's
spatial chunk shape, this computes - with pure arithmetic, no I/O - everything needed to
consolidate many shards into one MDIO store safely and with maximum parallelism:

  * which output chunks each shard writes,
  * per-chunk write mode: ``"fill"`` (single owner -> fast pure write) vs ``"merge"``
    (multiple owners -> read-modify-write so shards sharing a boundary chunk don't clobber
    each other),
  * the shard conflict graph (two shards conflict iff they share a chunk), and
  * a deterministic set of concurrency ``waves`` (no two conflicting shards in the same
    wave), so an external orchestrator can parallelize non-overlapping shards and stage
    the conflicting ones.

Orchestration itself is intentionally NOT handled here (or anywhere in MDIO) - the caller
turns this plan into an execution graph for whatever runtime it uses.

Determinism: outputs are a pure function of the inputs (coverage, chunk shape, and the
shard iteration order used for tie-breaking in wave coloring).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

# chunk index in the chunk grid (one int per spatial dim)
ChunkIndex = tuple[int, ...]
# a shard's global coverage: spatial dim name -> (start, stop) half-open index range
Coverage = dict[str, tuple[int, int]]


@dataclass
class ConsolidationPlan:
    """Deterministic consolidation plan (see module docstring)."""

    spatial_dims: tuple[str, ...]
    chunk_shape: tuple[int, ...]
    chunk_owners: dict[ChunkIndex, list[str]]
    chunk_modes: dict[ChunkIndex, str]
    shard_chunks: dict[str, list[ChunkIndex]]
    shard_merge_chunks: dict[str, list[ChunkIndex]]
    conflicts: dict[str, list[str]]
    waves: list[list[str]]
    global_shape: tuple[int, ...] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def merge_chunk_count(self) -> int:
        """Number of chunks that require read-modify-write."""
        return sum(1 for m in self.chunk_modes.values() if m == "merge")

    def merge_chunks_for(self, shard_id: str) -> set[ChunkIndex]:
        """Set of chunks this shard must write in ``merge`` (RMW) mode."""
        return set(self.shard_merge_chunks.get(shard_id, []))

    def to_dict(self) -> dict:
        """JSON-serializable view (tuple keys rendered as comma-joined strings)."""

        def _key(idx: ChunkIndex) -> str:
            return ",".join(map(str, idx))

        return {
            "spatial_dims": list(self.spatial_dims),
            "chunk_shape": list(self.chunk_shape),
            "global_shape": list(self.global_shape) if self.global_shape else None,
            "chunk_owners": {_key(k): v for k, v in self.chunk_owners.items()},
            "chunk_modes": {_key(k): v for k, v in self.chunk_modes.items()},
            "shard_chunks": {s: [_key(c) for c in cs] for s, cs in self.shard_chunks.items()},
            "shard_merge_chunks": {s: [_key(c) for c in cs] for s, cs in self.shard_merge_chunks.items()},
            "conflicts": self.conflicts,
            "waves": self.waves,
            "merge_chunk_count": self.merge_chunk_count,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConsolidationPlan:
        """Reconstruct a plan from :meth:`to_dict` output (inverse of ``to_dict``).

        Chunk keys serialized as comma-joined strings (e.g. ``"1,0"``) are parsed back to
        integer tuples. Derived-only fields (``merge_chunk_count``) are ignored.
        """

        def _idx(key: str) -> ChunkIndex:
            return tuple(int(p) for p in key.split(",")) if key else ()

        return cls(
            spatial_dims=tuple(data["spatial_dims"]),
            chunk_shape=tuple(int(c) for c in data["chunk_shape"]),
            chunk_owners={_idx(k): list(v) for k, v in data.get("chunk_owners", {}).items()},
            chunk_modes={_idx(k): v for k, v in data.get("chunk_modes", {}).items()},
            shard_chunks={s: [_idx(c) for c in cs] for s, cs in data.get("shard_chunks", {}).items()},
            shard_merge_chunks={s: [_idx(c) for c in cs] for s, cs in data.get("shard_merge_chunks", {}).items()},
            conflicts={s: list(v) for s, v in data.get("conflicts", {}).items()},
            waves=[list(w) for w in data.get("waves", [])],
            global_shape=tuple(data["global_shape"]) if data.get("global_shape") else None,
            warnings=list(data.get("warnings", [])),
        )


def coverage_from_index_arrays(
    global_coords: Mapping[str, Sequence],
    shard_values: Mapping[str, Sequence],
) -> Coverage:
    """Compute a shard's global index coverage from its coordinate values.

    Args:
        global_coords: dim name -> sorted global coordinate vector (the union across shards).
        shard_values: dim name -> the coordinate values present in this shard.

    Returns:
        dim name -> (start, stop) half-open global index range for the shard.
    """
    import numpy as np

    coverage: Coverage = {}
    for dim, gcoords in global_coords.items():
        if dim not in shard_values:
            continue
        g = np.asarray(gcoords)
        idx = np.searchsorted(g, np.asarray(shard_values[dim]))
        coverage[dim] = (int(idx.min()), int(idx.max()) + 1)
    return coverage


def _chunk_indices_for(coverage: Coverage, spatial_dims: Sequence[str], chunk_shape: Sequence[int]) -> list[ChunkIndex]:
    """All chunk-grid indices a coverage box touches."""
    per_dim_ranges = []
    for dim, csize in zip(spatial_dims, chunk_shape, strict=True):
        start, stop = coverage[dim]
        first = start // csize
        last = (stop - 1) // csize
        per_dim_ranges.append(range(first, last + 1))
    return list(itertools.product(*per_dim_ranges))


def _color_waves(shard_ids: Sequence[str], conflicts: dict[str, list[str]]) -> list[list[str]]:
    """Greedy deterministic graph coloring -> concurrency waves.

    No two conflicting shards share a wave, so within a wave no two shards write the same
    chunk. Iteration follows ``shard_ids`` order for reproducibility.
    """
    color: dict[str, int] = {}
    for sid in shard_ids:
        used = {color[n] for n in conflicts.get(sid, []) if n in color}
        c = 0
        while c in used:
            c += 1
        color[sid] = c

    num_waves = (max(color.values()) + 1) if color else 0
    waves: list[list[str]] = [[] for _ in range(num_waves)]
    for sid in shard_ids:
        waves[color[sid]].append(sid)
    return waves


def plan_consolidation(
    shard_coverage: Mapping[str, Coverage],
    spatial_dims: Sequence[str],
    chunk_shape: Sequence[int],
    global_shape: Sequence[int] | None = None,
) -> ConsolidationPlan:
    """Compute a deterministic consolidation plan.

    Args:
        shard_coverage: shard id -> {spatial dim -> (start, stop)} global index coverage.
            Iteration order of this mapping is the tie-break order for wave coloring.
        spatial_dims: Ordered spatial dimension names (must match the store's dim order,
            excluding the trailing sample dimension).
        chunk_shape: Spatial chunk sizes, aligned to ``spatial_dims``. MUST equal the
            store's data-variable spatial chunk sizes for the write-mode mask to line up.
        global_shape: Optional spatial global sizes (for validation/reporting).

    Returns:
        A :class:`ConsolidationPlan`.

    Raises:
        ValueError: If a shard is missing coverage for a spatial dimension.
    """
    spatial_dims = tuple(spatial_dims)
    chunk_shape = tuple(int(c) for c in chunk_shape)
    warnings: list[str] = []

    shard_ids = list(shard_coverage.keys())

    shard_chunks: dict[str, list[ChunkIndex]] = {}
    chunk_owners: dict[ChunkIndex, list[str]] = {}
    for sid in shard_ids:
        cov = shard_coverage[sid]
        missing = [d for d in spatial_dims if d not in cov]
        if missing:
            err = f"Shard '{sid}' missing coverage for dimensions {missing}."
            raise ValueError(err)
        chunks = _chunk_indices_for(cov, spatial_dims, chunk_shape)
        shard_chunks[sid] = chunks
        for c in chunks:
            chunk_owners.setdefault(c, []).append(sid)

    chunk_modes: dict[ChunkIndex, str] = {
        c: ("merge" if len(owners) > 1 else "fill") for c, owners in chunk_owners.items()
    }

    shard_merge_chunks: dict[str, list[ChunkIndex]] = {
        sid: [c for c in chunks if chunk_modes[c] == "merge"] for sid, chunks in shard_chunks.items()
    }

    # Conflict graph: shards sharing any chunk (necessarily a merge chunk).
    conflicts: dict[str, list[str]] = {sid: [] for sid in shard_ids}
    for owners in chunk_owners.values():
        if len(owners) < 2:
            continue
        for a in owners:
            for b in owners:
                if a != b and b not in conflicts[a]:
                    conflicts[a].append(b)

    if any(chunk_modes[c] == "merge" for c in chunk_modes):
        warnings.append(
            "Some chunks are shared by multiple shards and will use read-modify-write. "
            "Shards sharing a chunk are placed in different waves and MUST run sequentially "
            "with respect to each other."
        )

    waves = _color_waves(shard_ids, conflicts)

    return ConsolidationPlan(
        spatial_dims=spatial_dims,
        chunk_shape=chunk_shape,
        chunk_owners=chunk_owners,
        chunk_modes=chunk_modes,
        shard_chunks=shard_chunks,
        shard_merge_chunks=shard_merge_chunks,
        conflicts=conflicts,
        waves=waves,
        global_shape=tuple(int(s) for s in global_shape) if global_shape is not None else None,
        warnings=warnings,
    )
