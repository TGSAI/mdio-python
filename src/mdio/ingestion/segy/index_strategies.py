"""Composable index strategies for transforming SEG-Y headers into indexable dimensions.

This module replaces the monolithic `GridOverrider` command dispatch with a small set of
single-responsibility `IndexStrategy` objects that can be composed via `CompositeStrategy`.

Strategies are selected by `IndexStrategyRegistry` from the typed `GridOverrides`
configuration plus optional template hints, preserving end-to-end ingestion behavior.
"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib import recfunctions as rfn

from mdio.core import Dimension
from mdio.ingestion.segy.header_analysis import ShotGunGeometryType
from mdio.ingestion.segy.header_analysis import StreamerShotGeometryType
from mdio.ingestion.segy.header_analysis import analyze_lines_for_guns
from mdio.ingestion.segy.header_analysis import analyze_non_indexed_headers
from mdio.ingestion.segy.header_analysis import analyze_streamer_headers
from mdio.ingestion.segy.schema_effects import CollapseToTraceEffect
from mdio.ingestion.segy.schema_effects import InsertTraceDimEffect
from mdio.segy.exceptions import GridOverrideKeysError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy.typing import DTypeLike
    from segy.arrays import HeaderArray

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.ingestion.schema.models import SchemaEffect
    from mdio.segy.geometry import GridOverrides

logger = logging.getLogger(__name__)


def append_header_field(headers: HeaderArray, name: str, values: np.ndarray) -> HeaderArray:
    """Append a per-trace field to a header array.

    ``.base`` is None for non-view arrays; fall back to the array itself.
    """
    base = headers.base if headers.base is not None else headers
    return rfn.append_fields(base, name, values, usemask=False)


def rank_within_groups(values: np.ndarray, group_keys: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the 0-based rank of each value within its group, plus a duplicate mask.

    Ranks are positional offsets within each group after sorting by group then value.
    They are dense only when the duplicate mask is all ``False``.

    Args:
        values: Per-trace values to rank.
        group_keys: Per-trace arrays whose combination identifies a trace's group.

    Returns:
        Ranks and a boolean duplicate mask, both aligned with ``values``.
    """
    num_traces = len(values)
    order = np.lexsort((values, *reversed(group_keys)))

    starts_group = np.zeros(num_traces, dtype=bool)
    if num_traces:
        starts_group[0] = True
    for key in group_keys:
        sorted_key = key[order]
        starts_group[1:] |= sorted_key[1:] != sorted_key[:-1]

    positions = np.arange(num_traces)
    group_start = np.maximum.accumulate(np.where(starts_group, positions, 0))
    ranks = np.empty(num_traces, dtype=np.uint32)
    ranks[order] = positions - group_start

    sorted_values = values[order]
    duplicates = np.zeros(num_traces, dtype=bool)
    duplicates[order[1:]] = ~starts_group[1:] & (sorted_values[1:] == sorted_values[:-1])
    return ranks, duplicates


def _binned_spatial_dims(template: AbstractDatasetTemplate | None) -> tuple[str, ...]:
    """Spatial dimensions that come from headers, excluding calculated ones."""
    if template is None:
        return ()
    calculated = set(template.calculated_dimension_names)
    return tuple(name for name in template.spatial_dimension_names if name not in calculated)


class IndexStrategy(ABC):
    """Abstract base for header indexing strategies.

    A strategy transforms a raw header array (e.g., adding or rebasing fields) and
    computes the resulting `Dimension` list. Strategies are composable through
    `CompositeStrategy`. The default `compute_dimensions` builds dimensions from unique
    header values; subclasses override only when they need different semantics
    (currently just `CompositeStrategy`).

    Subclasses with header preconditions set `required_keys` so the ingestion reader and
    `CompositeStrategy` can raise `GridOverrideKeysError` with a clear
    "missing fields" message before NumPy fails on a deeper key lookup.
    """

    @property
    def required_keys(self) -> frozenset[str]:
        """Header field names that must be present before `transform_headers` runs.

        Empty by default. Override on subclasses whose transform indexes specific fields.
        """
        return frozenset()

    def validate_headers(self, headers: HeaderArray) -> None:
        """Raise `GridOverrideKeysError` if any required header field is missing.

        Callers (the ingestion reader and `CompositeStrategy`) invoke this before each
        transform so failure points at the user-facing override name rather than at a NumPy
        structured-array key error.
        """
        required = self.required_keys
        if not required:
            return
        present = set(headers.dtype.names or ())
        if not required.issubset(present):
            raise GridOverrideKeysError(self.name, set(required))

    @abstractmethod
    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Return a new header array with this strategy's transformation applied."""

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Build one `Dimension` per requested name from unique header values.

        Names absent from `headers.dtype.names` are silently skipped.
        """
        return [
            Dimension(coords=np.unique(headers[name]), name=name) for name in dim_names if name in headers.dtype.names
        ]

    def schema_effect(self) -> SchemaEffect | None:
        """Schema reshape this strategy implies, or ``None`` if it leaves the layout unchanged.

        Most strategies only transform headers. Only strategies that introduce a ``trace``
        dimension the template did not declare (see :class:`DuplicateHandlingStrategy` and
        :class:`NonBinnedStrategy`) reshape the resolved schema. Co-locating the reshape with
        the header transform keeps the two views of an override from drifting.
        """
        return None

    @property
    def name(self) -> str:
        """Return the strategy's class name; useful for logging and tests."""
        return self.__class__.__name__


class RegularGridStrategy(IndexStrategy):
    """Default strategy: headers untouched, dimensions are unique values per name."""

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Pass headers through unchanged."""
        return headers


class DuplicateHandlingStrategy(IndexStrategy):
    """Disambiguate duplicate index tuples by appending a per-tuple `trace` counter.

    Counts occurrences of each unique combination of dimension fields (excluding
    coordinate fields and any caller-declared `excluded_fields`), then attaches the
    resulting 1-based counter as a new `trace` field on the original headers.

    Args:
        coord_fields: Names of header fields that are template coordinates and must be
            excluded from the dimension grouping (their values vary independently of the
            grid index).
        excluded_fields: Additional fields to exclude from grouping. Used by
            `NonBinnedStrategy` to keep the explicit `non_binned_dims` from
            polluting the per-tuple counter.
        dtype: NumPy dtype for the appended `trace` counter.
    """

    def __init__(
        self,
        coord_fields: Iterable[str] = (),
        excluded_fields: Iterable[str] = (),
        dtype: DTypeLike = np.int16,
    ) -> None:
        self.coord_fields = frozenset(coord_fields)
        self.excluded_fields = frozenset(excluded_fields)
        self.dtype = dtype

    def _dim_fields(self, headers: HeaderArray) -> list[str]:
        """Header field names that participate in the duplicate grouping."""
        return [
            name
            for name in headers.dtype.names
            if name != "trace" and name not in self.coord_fields and name not in self.excluded_fields
        ]

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Append a per-dimension-tuple `trace` counter to headers."""
        dim_fields = self._dim_fields(headers)
        dim_headers = headers[dim_fields] if dim_fields else headers
        with_trace = analyze_non_indexed_headers(dim_headers, dtype=self.dtype)

        if with_trace is None or "trace" not in with_trace.dtype.names:
            return headers

        trace_values = np.array(with_trace["trace"])
        return rfn.append_fields(headers, "trace", trace_values, usemask=False)

    def schema_effect(self) -> SchemaEffect:
        """Insert a chunksize-1 ``trace`` dimension to disambiguate duplicate index tuples."""
        return InsertTraceDimEffect(chunksize=1)


class NonBinnedStrategy(DuplicateHandlingStrategy):
    """Collapse selected non-binned dimensions into a single `trace` dimension.

    Inherits the per-tuple `trace` counter from `DuplicateHandlingStrategy`, excluding the
    collapsed dims from the grouping so the counter only varies along the remaining dims, and
    owns the matching schema reshape (`CollapseToTraceEffect`). Both views of the override --
    the header transform and the schema layout, including the `trace` chunk size -- are
    therefore defined together here.

    Args:
        chunksize: Chunk size assigned to the inserted `trace` dimension by the schema effect.
        non_binned_dims: Header fields collapsed into `trace`. They are excluded from
            the duplicate grouping so the counter only varies along the remaining dims.
        coord_fields: Template coordinate names to exclude from grouping.
        dtype: NumPy dtype for the appended `trace` counter.
    """

    def __init__(
        self,
        chunksize: int,
        non_binned_dims: Iterable[str],
        coord_fields: Iterable[str] = (),
        dtype: DTypeLike = np.int16,
    ) -> None:
        collapse_dims = tuple(non_binned_dims)
        super().__init__(
            coord_fields=coord_fields,
            excluded_fields=collapse_dims,
            dtype=dtype,
        )
        self._chunksize = chunksize
        self._collapse_dims = collapse_dims

    def schema_effect(self) -> SchemaEffect:
        """Collapse the non-binned dims into a ``trace`` dimension sized by ``chunksize``."""
        return CollapseToTraceEffect(chunksize=self._chunksize, collapse_dims=self._collapse_dims)


class ChannelWrappingStrategy(IndexStrategy):
    """Renumber streamer channels per cable when geometry is Type B.

    Detects whether channel numbering is per-cable (Type A; pass-through) or sequential
    across cables (Type B; rebase to 1..N per cable).
    """

    @property
    def required_keys(self) -> frozenset[str]:
        """Streamer channel detection needs the cable-channel-shot triplet."""
        return frozenset({"shot_point", "cable", "channel"})

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Rebase `channel` per cable for Type B geometry; pass through for Type A."""
        unique_cables, cable_chan_min, cable_chan_max, geom_type = analyze_streamer_headers(headers)

        logger.info("Ingesting dataset as %s", geom_type.name)
        for cable, chan_min, chan_max in zip(unique_cables, cable_chan_min, cable_chan_max, strict=True):
            logger.info("Cable: %s has min chan: %s and max chan: %s", cable, chan_min, chan_max)

        if geom_type != StreamerShotGeometryType.B:
            return headers

        for idx, cable in enumerate(unique_cables):
            cable_idxs = np.where(headers["cable"][:] == cable)
            headers["channel"][cable_idxs] = headers["channel"][cable_idxs] - cable_chan_min[idx] + 1

        return headers


class ShotWrappingStrategy(IndexStrategy):
    """Derive a dense `shot_index` field from sparse or interleaved `shot_point` values.

    The two configurations differ in:

    * `line_field` -- `sail_line` for streamer, `shot_line` for OBN.
    * `always_calculate` -- streamer skips the transform entirely for Type A geometries
      (per-gun shot points are already dense), OBN always emits `shot_index` because the
      template declares it as a calculated dimension.

    Args:
        line_field: Header field used to group shots into independent lines.
        always_calculate: When `True`, emit `shot_index` for every geometry type. For
            Type A this builds a 0-based `np.searchsorted` over sorted unique shot
            points per line.
    """

    _STREAMER_LINE_FIELD = "sail_line"

    def __init__(self, line_field: str, always_calculate: bool = False) -> None:
        self.line_field = line_field
        self.always_calculate = always_calculate

    @property
    def required_keys(self) -> frozenset[str]:
        """Streamer (`sail_line`) needs cable and channel too; OBN (`shot_line`) does not."""
        base = {self.line_field, "gun", "shot_point"}
        if self.line_field == self._STREAMER_LINE_FIELD:
            base |= {"cable", "channel"}
        return frozenset(base)

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Append `shot_index` derived from `shot_point` per line."""
        unique_lines, unique_guns_per_line, geom_type = analyze_lines_for_guns(headers, line_field=self.line_field)

        logger.info("Ingesting dataset as shot type: %s (line_field=%s)", geom_type.name, self.line_field)

        max_num_guns = 1
        for line_val in unique_lines:
            guns = unique_guns_per_line[str(line_val)]
            logger.info("%s: %s has guns: %s", self.line_field, line_val, guns)
            max_num_guns = max(len(guns), max_num_guns)

        if geom_type == ShotGunGeometryType.A and not self.always_calculate:
            return headers

        shot_index = np.empty(len(headers), dtype="uint32")
        headers = append_header_field(headers, "shot_index", shot_index)

        if geom_type == ShotGunGeometryType.B:
            for line_val in unique_lines:
                line_idxs = np.where(headers[self.line_field][:] == line_val)
                headers["shot_index"][line_idxs] = np.floor(headers["shot_point"][line_idxs] / max_num_guns)
                headers["shot_index"][line_idxs] -= headers["shot_index"][line_idxs].min()
        else:
            for line_val in unique_lines:
                line_idxs = np.where(headers[self.line_field][:] == line_val)
                shot_points = headers["shot_point"][line_idxs]
                unique_shots = np.unique(shot_points)
                headers["shot_index"][line_idxs] = np.searchsorted(unique_shots, shot_points)

        return headers


class HeaderRankingStrategy(IndexStrategy):
    """Derive a dense 0-based dimension by ranking a header field within groups of traces.

    Within each ``group_fields`` group, a trace's index is the position of its
    ``value_field`` in that group's sorted unique values. The source values are never
    modified; templates typically keep the original header as a coordinate.

    Args:
        value_field: Header field to rank (e.g. ``epoch``).
        index_name: Name of the appended dimension field (e.g. ``segment_index``).
        group_fields: Header fields identifying the ranking group. Must be non-empty.

    Raises:
        ValueError: If ``group_fields`` is empty.
    """

    def __init__(
        self,
        value_field: str,
        index_name: str,
        group_fields: Iterable[str],
    ) -> None:
        self.value_field = value_field
        self.index_name = index_name
        self.group_fields = tuple(group_fields)
        if not self.group_fields:
            msg = (
                f"HeaderRankingStrategy for {index_name!r} requires non-empty group_fields; "
                "global ranking is not supported."
            )
            raise ValueError(msg)

    @property
    def required_keys(self) -> frozenset[str]:
        """The ranked field plus every field that defines a ranking group."""
        return frozenset({self.value_field, *self.group_fields})

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Append `index_name`, the rank of `value_field` within each `group_fields` group."""
        values = np.asarray(headers[self.value_field])
        group_keys = tuple(np.asarray(headers[name]) for name in self.group_fields)
        ranks, duplicates = rank_within_groups(values, group_keys)

        if duplicates.any():
            raise ValueError(self._duplicate_message(values, group_keys, duplicates))

        headers = append_header_field(headers, self.index_name, ranks)

        logger.info(
            "Ranked '%s' into dense '%s' across %d group(s) keyed by %s",
            self.value_field,
            self.index_name,
            int(np.count_nonzero(ranks == 0)),
            self.group_fields,
        )
        return headers

    def _duplicate_message(
        self,
        values: np.ndarray,
        group_keys: tuple[np.ndarray, ...],
        duplicates: np.ndarray,
    ) -> str:
        """Describe the first offending key. Only reached on the failure path, so an extra scan is fine."""
        offender = int(np.flatnonzero(duplicates)[0])
        key_fields = (*self.group_fields, self.value_field)

        matches = values == values[offender]
        for key in group_keys:
            matches &= key == key[offender]

        key_desc = ", ".join(
            f"{name}={key[offender]}" for name, key in zip(key_fields, (*group_keys, values), strict=True)
        )
        return (
            f"Duplicate {self.value_field!r} for ranking into {self.index_name!r}: "
            f"{key_desc} appears {int(np.count_nonzero(matches))} times. "
            f"Each ({', '.join(key_fields)}) combination must be unique."
        )


class ComponentSynthesisStrategy(IndexStrategy):
    """Synthesize template-required dimension fields that are absent from the headers.

    Currently used to fill the `component` dimension with a constant value of 1 for
    OBN templates whose SEG-Y spec does not include a component header.

    Args:
        synthesize_dims: Names of dimension fields to synthesize when missing.
    """

    def __init__(self, synthesize_dims: Iterable[str]) -> None:
        self.synthesize_dims = tuple(synthesize_dims)

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Append constant-1 fields for any synthesize_dims not already present."""
        for dim in self.synthesize_dims:
            if dim in headers.dtype.names:
                continue
            logger.warning(
                "SEG-Y headers do not contain '%s' field required by template; "
                "synthesizing dimension with constant value 1 for all traces.",
                dim,
            )
            comp_array = np.ones(len(headers), dtype=np.uint8)
            headers = append_header_field(headers, dim, comp_array)
        return headers


class CompositeStrategy(IndexStrategy):
    """Apply multiple strategies in order; each transform feeds the next.

    Dimension computation is delegated to the final strategy on the assumption it is
    aware of all preceding header transformations.
    """

    def __init__(self, strategies: list[IndexStrategy]) -> None:
        if not strategies:
            msg = "CompositeStrategy requires at least one strategy"
            raise ValueError(msg)
        self.strategies = strategies

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Validate then run each child strategy's transform in sequence.

        Each step re-validates against the running header array, so a strategy that
        produces a field (e.g. `ComponentSynthesisStrategy` adding `component`)
        can satisfy a later strategy's `required_keys`.
        """
        result = headers
        for strategy in self.strategies:
            logger.debug("Applying strategy: %s", strategy.name)
            strategy.validate_headers(result)
            result = strategy.transform_headers(result)
        return result

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Delegate to the final child strategy."""
        return self.strategies[-1].compute_dimensions(headers, dim_names)

    def schema_effect(self) -> SchemaEffect | None:
        """Return the single child reshape; at most one composed strategy produces one."""
        for strategy in self.strategies:
            effect = strategy.schema_effect()
            if effect is not None:
                return effect
        return None


class IndexStrategyRegistry:
    """Picks the right `IndexStrategy` from grid overrides and template hints.

    The registry maps a `GridOverrides` to the header-transforming `IndexStrategy` in one
    place (`create_strategy`). The schema-reshaping `SchemaEffect` is not selected by a
    second switch: it is read off that same strategy (`schema_effect`), so the header view
    and the schema view of an override cannot drift.
    """

    def schema_effect(
        self,
        grid_overrides: GridOverrides | None,
        template: AbstractDatasetTemplate | None = None,
    ) -> SchemaEffect | None:
        """Return the schema reshaping implied by `grid_overrides`, if any.

        Derived from the same strategy that will transform headers, so layout changes stay in
        lock-step with the header transform.

        Args:
            grid_overrides: Typed grid override configuration, or `None`.
            template: Template the overrides were validated against. Required by overrides
                that read dimension names off the template, such as `calculate_segment_index`.

        Returns:
            The matching `SchemaEffect`, or `None` when no layout change applies.
        """
        if not grid_overrides:
            return None
        return self.create_strategy(grid_overrides, template=template).schema_effect()

    def create_strategy(
        self,
        grid_overrides: GridOverrides | None = None,
        synthesize_dims: tuple[str, ...] = (),
        template: AbstractDatasetTemplate | None = None,
    ) -> IndexStrategy:
        """Build a strategy (possibly composite) for the given config.

        Strategy ordering, when multiple flags are set:

        1. `ComponentSynthesisStrategy` (so later strategies can rely on the synthesized
           field being present).
        2. `ChannelWrappingStrategy` (rebases `channel` before any shot calculation).
        3. `ShotWrappingStrategy` for `auto_shot_wrap` (streamer; `sail_line`).
        4. `ShotWrappingStrategy` for `calculate_shot_index` (OBN; `shot_line`,
           `always_calculate=True`).
        5. `HeaderRankingStrategy` for `calculate_segment_index` (ranks `epoch` into
           `segment_index`).
        6. `NonBinnedStrategy` or `DuplicateHandlingStrategy` (mutually exclusive;
           `non_binned` wins when both are set).

        Args:
            grid_overrides: Typed grid override configuration, or `None` for no
                user-driven overrides.
            synthesize_dims: Dimensions to synthesize if missing (e.g., `component`).
            template: Optional dataset template; used to look up coordinate names so
                duplicate-handling counters group on dimension fields only, and to derive
                ranking groups for `calculate_segment_index`.

        Returns:
            A single `IndexStrategy` instance. Returns `RegularGridStrategy` when no
            overrides and no synthesis are required.
        """
        strategies: list[IndexStrategy] = []

        if synthesize_dims:
            strategies.append(ComponentSynthesisStrategy(synthesize_dims))

        coord_fields: tuple[str, ...] = template.coordinate_names if template is not None else ()

        if grid_overrides:
            if grid_overrides.auto_channel_wrap:
                strategies.append(ChannelWrappingStrategy())

            if grid_overrides.auto_shot_wrap:
                strategies.append(ShotWrappingStrategy(line_field="sail_line", always_calculate=False))

            if grid_overrides.calculate_shot_index:
                strategies.append(ShotWrappingStrategy(line_field="shot_line", always_calculate=True))

            if grid_overrides.calculate_segment_index:
                strategies.append(
                    HeaderRankingStrategy(
                        value_field="epoch",
                        index_name="segment_index",
                        group_fields=_binned_spatial_dims(template),
                    )
                )

            if grid_overrides.non_binned:
                strategies.append(
                    NonBinnedStrategy(
                        chunksize=grid_overrides.chunksize,
                        non_binned_dims=grid_overrides.non_binned_dims or (),
                        coord_fields=coord_fields,
                    )
                )
            elif grid_overrides.has_duplicates:
                strategies.append(DuplicateHandlingStrategy(coord_fields=coord_fields))

        if not strategies:
            return RegularGridStrategy()
        if len(strategies) == 1:
            return strategies[0]
        return CompositeStrategy(strategies)
