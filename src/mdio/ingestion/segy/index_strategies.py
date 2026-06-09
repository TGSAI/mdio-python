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

    from numpy.typing import DTypeLike
    from segy.arrays import HeaderArray

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.ingestion.schema.models import SchemaEffect
    from mdio.segy.geometry import GridOverrides

logger = logging.getLogger(__name__)


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
        # `.base` is None for non-view arrays; fall back to the array itself.
        base_array = headers.base if headers.base is not None else headers
        headers = rfn.append_fields(base_array, "shot_index", shot_index, usemask=False)

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
            base_array = headers.base if headers.base is not None else headers
            headers = rfn.append_fields(base_array, dim, comp_array, usemask=False)
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

    def schema_effect(self, grid_overrides: GridOverrides | None) -> SchemaEffect | None:
        """Return the schema reshaping implied by `grid_overrides`, if any.

        Derived from the same strategy that will transform headers, so layout changes stay in
        lock-step with the header transform. Template and synthesis hints do not affect the
        reshape, so they are omitted when building the strategy here.

        Args:
            grid_overrides: Typed grid override configuration, or `None`.

        Returns:
            The matching `SchemaEffect`, or `None` when no layout change applies.
        """
        if not grid_overrides:
            return None
        return self.create_strategy(grid_overrides).schema_effect()

    def create_strategy(
        self,
        grid_overrides: GridOverrides | None = None,
        synthesize_dims: tuple[str, ...] = (),
        template: AbstractDatasetTemplate | None = None,
    ) -> IndexStrategy:
        """Build a strategy (possibly composite) for the given config.

        Strategy ordering, when multiple flags are set, mirrors previous behavior:

        1. `ComponentSynthesisStrategy` (so later strategies can rely on the synthesized
           field being present).
        2. `ChannelWrappingStrategy` (rebases `channel` before any shot calculation).
        3. `ShotWrappingStrategy` for `auto_shot_wrap` (streamer; `sail_line`).
        4. `ShotWrappingStrategy` for `calculate_shot_index` (OBN; `shot_line`,
           `always_calculate=True`).
        5. `NonBinnedStrategy` or `DuplicateHandlingStrategy` (mutually exclusive;
           `non_binned` wins when both are set).

        Args:
            grid_overrides: Typed grid override configuration, or `None` for no
                user-driven overrides.
            synthesize_dims: Dimensions to synthesize if missing (e.g., `component`).
            template: Optional dataset template; used to look up coordinate names so
                duplicate-handling counters group on dimension fields only.

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
