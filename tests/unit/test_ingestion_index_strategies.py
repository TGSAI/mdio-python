"""Unit tests for the v1.2 ingestion index strategies and the strategy registry.

These tests exercise individual :class:`mdio.ingestion.index_strategies.IndexStrategy`
subclasses with synthetic structured numpy arrays (mimicking the shape semantics of
:class:`segy.arrays.HeaderArray`) so they remain fast and do not require any real SEG-Y
data.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdio.builder.template_registry import TemplateRegistry
from mdio.ingestion.index_strategies import ChannelWrappingStrategy
from mdio.ingestion.index_strategies import ComponentSynthesisStrategy
from mdio.ingestion.index_strategies import CompositeStrategy
from mdio.ingestion.index_strategies import DuplicateHandlingStrategy
from mdio.ingestion.index_strategies import IndexStrategyRegistry
from mdio.ingestion.index_strategies import NonBinnedStrategy
from mdio.ingestion.index_strategies import RegularGridStrategy
from mdio.ingestion.index_strategies import ShotWrappingStrategy
from mdio.segy.exceptions import GridOverrideKeysError
from mdio.segy.geometry import GridOverrider
from mdio.segy.geometry import GridOverrides


def _make_struct(data: dict[str, np.ndarray]) -> np.ndarray:
    """Build a 1-D structured array from a name -> 1-D array mapping."""
    names = list(data.keys())
    arrays = [data[name] for name in names]
    n = len(arrays[0])
    dtype = np.dtype([(name, arr.dtype) for name, arr in zip(names, arrays, strict=True)])
    out = np.empty(n, dtype=dtype)
    for name, arr in zip(names, arrays, strict=True):
        out[name] = arr
    return out


# ---------------------------------------------------------------------------
# IndexStrategyRegistry
# ---------------------------------------------------------------------------


class TestIndexStrategyRegistry:
    """Selection rules for :class:`IndexStrategyRegistry.create_strategy`."""

    def test_default_returns_regular_grid(self) -> None:
        """No grid overrides and no synthesis -> regular grid."""
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=None)
        assert isinstance(strategy, RegularGridStrategy)

    def test_falsy_overrides_returns_regular_grid(self) -> None:
        """A default ``GridOverrides()`` with all flags off must be treated as no-op."""
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=GridOverrides())
        assert isinstance(strategy, RegularGridStrategy)

    def test_synthesize_dims_only(self) -> None:
        """Synthesis-only configuration returns a single ComponentSynthesisStrategy."""
        strategy = IndexStrategyRegistry().create_strategy(synthesize_dims=("component",))
        assert isinstance(strategy, ComponentSynthesisStrategy)
        assert strategy.synthesize_dims == ("component",)

    def test_non_binned_only(self) -> None:
        """``non_binned`` -> NonBinnedStrategy with chunksize and excluded dims wired."""
        overrides = GridOverrides(non_binned=True, chunksize=64, non_binned_dims=["channel"])
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=overrides)
        assert isinstance(strategy, NonBinnedStrategy)
        assert strategy.chunksize == 64
        assert strategy.non_binned_dims == ("channel",)

    def test_has_duplicates_only(self) -> None:
        """``has_duplicates`` -> DuplicateHandlingStrategy."""
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=GridOverrides(has_duplicates=True))
        assert isinstance(strategy, DuplicateHandlingStrategy)

    def test_non_binned_wins_over_has_duplicates(self) -> None:
        """Both flags set -> NonBinned wins (matches v1.x semantics)."""
        overrides = GridOverrides(
            non_binned=True,
            chunksize=8,
            non_binned_dims=["channel"],
            has_duplicates=True,
        )
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=overrides)
        assert isinstance(strategy, NonBinnedStrategy)

    def test_composite_with_channel_wrap(self) -> None:
        """Channel wrap + non-binned -> CompositeStrategy ordered for safe layering."""
        overrides = GridOverrides(
            auto_channel_wrap=True,
            non_binned=True,
            chunksize=64,
            non_binned_dims=["channel"],
        )
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=overrides)
        assert isinstance(strategy, CompositeStrategy)
        assert [s.name for s in strategy.strategies] == ["ChannelWrappingStrategy", "NonBinnedStrategy"]

    def test_synthesize_dims_runs_first(self) -> None:
        """Synthesis must run before any strategy that may depend on the synthesized field."""
        overrides = GridOverrides(calculate_shot_index=True)
        strategy = IndexStrategyRegistry().create_strategy(
            grid_overrides=overrides,
            synthesize_dims=("component",),
        )
        assert isinstance(strategy, CompositeStrategy)
        assert strategy.strategies[0].name == "ComponentSynthesisStrategy"

    def test_auto_shot_wrap_uses_sail_line(self) -> None:
        """``auto_shot_wrap`` is the streamer flag: line_field=sail_line, no always-calc."""
        overrides = GridOverrides(auto_shot_wrap=True)
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=overrides)
        assert isinstance(strategy, ShotWrappingStrategy)
        assert strategy.line_field == "sail_line"
        assert strategy.always_calculate is False

    def test_calculate_shot_index_uses_shot_line(self) -> None:
        """``calculate_shot_index`` is the OBN flag: line_field=shot_line, always-calc."""
        overrides = GridOverrides(calculate_shot_index=True)
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=overrides)
        assert isinstance(strategy, ShotWrappingStrategy)
        assert strategy.line_field == "shot_line"
        assert strategy.always_calculate is True

    def test_template_coord_names_propagate_to_duplicate_strategy(self) -> None:
        """Template coordinates flow into the duplicate-handling strategy as exclusions."""
        template = TemplateRegistry().get("StreamerShotGathers3D")
        strategy = IndexStrategyRegistry().create_strategy(
            grid_overrides=GridOverrides(has_duplicates=True),
            template=template,
        )
        assert isinstance(strategy, DuplicateHandlingStrategy)
        assert strategy.coord_fields == frozenset(template.coordinate_names)


# ---------------------------------------------------------------------------
# RegularGridStrategy
# ---------------------------------------------------------------------------


class TestRegularGridStrategy:
    """Pass-through strategy used when no overrides are required."""

    def test_returns_unique_dims(self) -> None:
        """Each dim name maps to its sorted unique values."""
        headers = _make_struct(
            {
                "inline": np.array([1, 1, 2, 2], dtype=np.int32),
                "crossline": np.array([10, 11, 10, 11], dtype=np.int32),
            }
        )
        dims = RegularGridStrategy().compute_dimensions(headers, ("inline", "crossline"))
        assert [d.name for d in dims] == ["inline", "crossline"]
        np.testing.assert_array_equal(dims[0].coords, [1, 2])
        np.testing.assert_array_equal(dims[1].coords, [10, 11])

    def test_unknown_dim_silently_skipped(self) -> None:
        """Names absent from the header dtype are dropped, matching v1.1 behavior."""
        headers = _make_struct({"inline": np.array([1, 2], dtype=np.int32)})
        dims = RegularGridStrategy().compute_dimensions(headers, ("inline", "missing"))
        assert [d.name for d in dims] == ["inline"]

    def test_default_required_keys_empty_and_validate_is_noop(self) -> None:
        """Strategies that don't override ``required_keys`` must not block any headers."""
        headers = _make_struct({"inline": np.array([1], dtype=np.int32)})
        strategy = RegularGridStrategy()
        assert strategy.required_keys == frozenset()
        strategy.validate_headers(headers)  # must not raise


# ---------------------------------------------------------------------------
# DuplicateHandlingStrategy
# ---------------------------------------------------------------------------


class TestDuplicateHandlingStrategy:
    """Counter-based ``trace`` field for duplicated index tuples."""

    def test_appends_per_tuple_counter(self) -> None:
        """Each (inline, crossline) tuple gets a 1-based duplicate counter."""
        headers = _make_struct(
            {
                "inline": np.array([1, 1, 1, 2], dtype=np.int32),
                "crossline": np.array([10, 10, 11, 10], dtype=np.int32),
            }
        )
        out = DuplicateHandlingStrategy().transform_headers(headers)
        assert "trace" in out.dtype.names
        # (1,10) appears twice -> {1, 2}; (1,11) once -> {1}; (2,10) once -> {1}.
        np.testing.assert_array_equal(np.sort(out["trace"]), [1, 1, 1, 2])

    def test_coord_fields_excluded_from_grouping(self) -> None:
        """Coordinate fields must not influence the duplicate counter."""
        headers = _make_struct(
            {
                "inline": np.array([1, 1, 2, 2], dtype=np.int32),
                "crossline": np.array([10, 10, 11, 11], dtype=np.int32),
                # 'cdp_x' varies per row but is a coord, so the counter must ignore it.
                "cdp_x": np.array([100.0, 100.5, 200.0, 200.5], dtype=np.float64),
            }
        )
        strategy = DuplicateHandlingStrategy(coord_fields=("cdp_x",))
        out = strategy.transform_headers(headers)
        # Each (inline, crossline) tuple appears twice -> counters cycle 1..2.
        np.testing.assert_array_equal(out["trace"], [1, 2, 1, 2])

    def test_excluded_fields_dropped_from_grouping(self) -> None:
        """Caller-declared excluded fields (e.g. non_binned_dims) are not part of the key."""
        headers = _make_struct(
            {
                "shot_point": np.array([1, 1, 1], dtype=np.int32),
                "channel": np.array([1, 2, 3], dtype=np.int32),
            }
        )
        strategy = DuplicateHandlingStrategy(excluded_fields=("channel",))
        out = strategy.transform_headers(headers)
        # Without excluding 'channel', counters would all be 1; excluding it groups by
        # shot_point alone so each row in the same shot_point gets a fresh counter.
        np.testing.assert_array_equal(out["trace"], [1, 2, 3])


# ---------------------------------------------------------------------------
# NonBinnedStrategy
# ---------------------------------------------------------------------------


class TestNonBinnedStrategy:
    """``NonBinned`` is the duplicate counter wired with explicit collapse dims."""

    def test_chunksize_and_dims_recorded(self) -> None:
        """Constructor stores both for the ``GridOverrider`` shim to use."""
        strategy = NonBinnedStrategy(chunksize=4, non_binned_dims=("channel",))
        assert strategy.chunksize == 4
        assert strategy.non_binned_dims == ("channel",)

    def test_collapse_dim_excluded_from_counter(self) -> None:
        """The non-binned dim must NOT participate in the duplicate counter grouping."""
        headers = _make_struct(
            {
                "shot_point": np.array([1, 1, 2, 2], dtype=np.int32),
                "cable": np.array([1, 2, 1, 2], dtype=np.int32),
                "channel": np.array([10, 11, 10, 11], dtype=np.int32),
            }
        )
        strategy = NonBinnedStrategy(chunksize=4, non_binned_dims=("channel",))
        out = strategy.transform_headers(headers)
        # Each (shot_point, cable) tuple appears once -> counters all equal 1.
        assert "trace" in out.dtype.names
        np.testing.assert_array_equal(out["trace"], [1, 1, 1, 1])

    def test_coord_fields_also_excluded(self) -> None:
        """Both template coords and non_binned_dims are removed from the grouping key."""
        headers = _make_struct(
            {
                "shot_point": np.array([1, 1, 2, 2], dtype=np.int32),
                "channel": np.array([1, 2, 1, 2], dtype=np.int32),
                "cdp_x": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
            }
        )
        strategy = NonBinnedStrategy(
            chunksize=4,
            non_binned_dims=("channel",),
            coord_fields=("cdp_x",),
        )
        out = strategy.transform_headers(headers)
        # Grouping is by shot_point only -> each shot_point has 2 rows -> counters {1, 2}.
        np.testing.assert_array_equal(out["trace"], [1, 2, 1, 2])


# ---------------------------------------------------------------------------
# ChannelWrappingStrategy
# ---------------------------------------------------------------------------


class TestChannelWrappingStrategy:
    """Streamer Type-A vs Type-B detection and channel rebasing."""

    def test_type_a_pass_through(self) -> None:
        """Type A (per-cable channel numbering with overlap) -> headers untouched."""
        headers = _make_struct(
            {
                "cable": np.array([1, 1, 2, 2], dtype=np.int32),
                "channel": np.array([1, 2, 1, 2], dtype=np.int32),
            }
        )
        out = ChannelWrappingStrategy().transform_headers(headers)
        np.testing.assert_array_equal(out["channel"], [1, 2, 1, 2])

    def test_type_b_renumbers_per_cable(self) -> None:
        """Type B (sequential numbering across cables) -> rebased to 1..N per cable."""
        headers = _make_struct(
            {
                "cable": np.array([1, 1, 2, 2], dtype=np.int32),
                "channel": np.array([1, 2, 3, 4], dtype=np.int32),
            }
        )
        out = ChannelWrappingStrategy().transform_headers(headers)
        # Cable 1: 1,2 -> 1,2; cable 2: 3,4 -> 1,2.
        np.testing.assert_array_equal(out["channel"], [1, 2, 1, 2])

    def test_required_keys(self) -> None:
        """Channel wrap declares the cable-channel-shot triplet as preconditions."""
        assert ChannelWrappingStrategy().required_keys == frozenset({"shot_point", "cable", "channel"})

    def test_validate_headers_raises_when_field_missing(self) -> None:
        """Missing ``cable`` -> :class:`GridOverrideKeysError`, not a deeper numpy crash."""
        headers = _make_struct(
            {
                "shot_point": np.array([1, 2], dtype=np.int32),
                "channel": np.array([1, 2], dtype=np.int32),
            }
        )
        strategy = ChannelWrappingStrategy()
        with pytest.raises(GridOverrideKeysError, match="ChannelWrappingStrategy"):
            strategy.validate_headers(headers)


# ---------------------------------------------------------------------------
# ShotWrappingStrategy
# ---------------------------------------------------------------------------


class TestShotWrappingStrategy:
    """Shot-index derivation for both streamer (sail_line) and OBN (shot_line)."""

    def test_type_b_streamer_emits_shot_index(self) -> None:
        """Sail line 1 with two interleaved guns -> dense shot_index per line."""
        headers = _make_struct(
            {
                "sail_line": np.array([1, 1, 1, 1], dtype=np.int32),
                "gun": np.array([1, 2, 1, 2], dtype=np.int32),
                "shot_point": np.array([1, 2, 3, 4], dtype=np.int32),
            }
        )
        out = ShotWrappingStrategy(line_field="sail_line").transform_headers(headers)
        assert "shot_index" in out.dtype.names
        # floor(shot_point / 2) zero-based per line: 0, 1, 1, 2.
        np.testing.assert_array_equal(out["shot_index"], [0, 1, 1, 2])

    def test_type_a_streamer_skipped_without_always_calculate(self) -> None:
        """Type A streamer geometry produces no shot_index unless always_calculate=True."""
        headers = _make_struct(
            {
                "sail_line": np.array([1, 1, 1, 1, 1, 1], dtype=np.int32),
                "gun": np.array([1, 1, 1, 2, 2, 2], dtype=np.int32),
                "shot_point": np.array([1, 2, 3, 1, 2, 3], dtype=np.int32),
            }
        )
        out = ShotWrappingStrategy(line_field="sail_line").transform_headers(headers)
        assert "shot_index" not in out.dtype.names

    def test_type_a_obn_always_calculates(self) -> None:
        """OBN forces shot_index calculation; Type A uses dense per-line searchsorted."""
        headers = _make_struct(
            {
                "shot_line": np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
                "gun": np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32),
                "shot_point": np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32),
            }
        )
        out = ShotWrappingStrategy(line_field="shot_line", always_calculate=True).transform_headers(headers)
        assert "shot_index" in out.dtype.names
        np.testing.assert_array_equal(out["shot_index"], [0, 1, 2, 3, 0, 1, 2, 3])

    def test_obn_multiline_type_a_processes_all_lines(self) -> None:
        """Regression: Type A detection on line 1 must not mask later lines."""
        headers = _make_struct(
            {
                "shot_line": np.array([1, 1, 2, 2, 3, 3], dtype=np.int32),
                "gun": np.array([1, 2, 1, 2, 1, 2], dtype=np.int32),
                "shot_point": np.array([1, 2, 1, 2, 1, 2], dtype=np.int32),
            }
        )
        out = ShotWrappingStrategy(line_field="shot_line", always_calculate=True).transform_headers(headers)
        # Each line gets independent dense per-line indices.
        np.testing.assert_array_equal(out["shot_index"], [0, 1, 0, 1, 0, 1])

    def test_required_keys_sail_line(self) -> None:
        """Streamer variant requires the streamer-cable-channel headers in addition to shot fields."""
        strategy = ShotWrappingStrategy(line_field="sail_line")
        assert strategy.required_keys == frozenset({"sail_line", "gun", "shot_point", "cable", "channel"})

    def test_required_keys_shot_line(self) -> None:
        """OBN variant deliberately omits cable/channel from required keys."""
        strategy = ShotWrappingStrategy(line_field="shot_line", always_calculate=True)
        assert strategy.required_keys == frozenset({"shot_line", "gun", "shot_point"})

    def test_validate_headers_raises_for_obn_when_missing_gun(self) -> None:
        """Missing ``gun`` on the OBN path -> :class:`GridOverrideKeysError`."""
        headers = _make_struct(
            {
                "shot_line": np.array([1, 1], dtype=np.int32),
                "shot_point": np.array([1, 2], dtype=np.int32),
            }
        )
        strategy = ShotWrappingStrategy(line_field="shot_line", always_calculate=True)
        with pytest.raises(GridOverrideKeysError, match="ShotWrappingStrategy"):
            strategy.validate_headers(headers)


# ---------------------------------------------------------------------------
# ComponentSynthesisStrategy
# ---------------------------------------------------------------------------


class TestComponentSynthesisStrategy:
    """Synthesize template-required dimensions when missing from the SEG-Y headers."""

    def test_synthesizes_missing_field(self) -> None:
        """Missing field is added with constant value 1 for every row."""
        headers = _make_struct({"receiver": np.array([1, 2, 3], dtype=np.int32)})
        out = ComponentSynthesisStrategy(("component",)).transform_headers(headers)
        assert "component" in out.dtype.names
        np.testing.assert_array_equal(out["component"], [1, 1, 1])

    def test_existing_field_left_alone(self) -> None:
        """If the field is already present, the existing values are preserved."""
        headers = _make_struct(
            {
                "receiver": np.array([1, 2, 3], dtype=np.int32),
                "component": np.array([2, 3, 4], dtype=np.uint8),
            }
        )
        out = ComponentSynthesisStrategy(("component",)).transform_headers(headers)
        np.testing.assert_array_equal(out["component"], [2, 3, 4])


# ---------------------------------------------------------------------------
# CompositeStrategy
# ---------------------------------------------------------------------------


class TestCompositeStrategy:
    """Strategy chaining with deterministic execution order."""

    def test_requires_at_least_one_strategy(self) -> None:
        """An empty composite is a programming error and must raise."""
        with pytest.raises(ValueError, match="at least one strategy"):
            CompositeStrategy([])

    def test_strategies_run_in_order(self) -> None:
        """Synthesis must produce the field that the next strategy then duplicates."""
        headers = _make_struct(
            {
                "shot_point": np.array([1, 1, 2, 2], dtype=np.int32),
                "channel": np.array([1, 2, 1, 2], dtype=np.int32),
            }
        )
        composite = CompositeStrategy(
            [
                ComponentSynthesisStrategy(("component",)),
                NonBinnedStrategy(chunksize=4, non_binned_dims=("channel",)),
            ]
        )
        out = composite.transform_headers(headers)
        assert "component" in out.dtype.names
        assert "trace" in out.dtype.names

    def test_progressive_validation_raises_for_first_unsatisfied_child(self) -> None:
        """Composite must surface a child's required-keys failure as :class:`GridOverrideKeysError`.

        Channel wrap needs ``cable``; ``RegularGridStrategy`` runs first and is a no-op,
        so the composite reaches channel wrap with the same incomplete headers and must
        raise rather than crash inside numpy.
        """
        headers = _make_struct(
            {
                "shot_point": np.array([1, 2], dtype=np.int32),
                "channel": np.array([1, 2], dtype=np.int32),
            }
        )
        composite = CompositeStrategy([RegularGridStrategy(), ChannelWrappingStrategy()])
        with pytest.raises(GridOverrideKeysError, match="ChannelWrappingStrategy"):
            composite.transform_headers(headers)


# ---------------------------------------------------------------------------
# GridOverrider template-compatibility checks (shim level)
# ---------------------------------------------------------------------------


class TestGridOverriderTemplateValidation:
    """Restore v1.1's template-type guards for shot-wrapping overrides.

    ``AutoShotWrap`` was streamer-only and ``CalculateShotIndex`` was OBN-only; pairing
    either with the wrong template silently produced incorrect shot indices. The shim
    raises :class:`TypeError` early so misconfigurations fail loudly at the API boundary.
    """

    def test_auto_shot_wrap_rejects_obn_template(self) -> None:
        """Streamer override + OBN template -> TypeError, no transform run."""
        headers = _make_struct(
            {
                "sail_line": np.array([1, 1], dtype=np.int32),
                "gun": np.array([1, 2], dtype=np.int32),
                "shot_point": np.array([1, 2], dtype=np.int32),
                "cable": np.array([1, 1], dtype=np.int32),
                "channel": np.array([1, 2], dtype=np.int32),
            }
        )
        template = TemplateRegistry().get("ObnReceiverGathers3D")
        with pytest.raises(TypeError, match="auto_shot_wrap.*Seismic3DStreamerFieldRecordsTemplate"):
            GridOverrider().run(
                headers,
                index_names=("sail_line", "gun", "shot_point"),
                grid_overrides={"AutoShotWrap": True},
                template=template,
            )

    def test_calculate_shot_index_rejects_streamer_template(self) -> None:
        """OBN override + streamer template -> TypeError."""
        headers = _make_struct(
            {
                "shot_line": np.array([1, 1], dtype=np.int32),
                "gun": np.array([1, 2], dtype=np.int32),
                "shot_point": np.array([1, 2], dtype=np.int32),
            }
        )
        template = TemplateRegistry().get("StreamerShotGathers3D")
        with pytest.raises(TypeError, match="calculate_shot_index.*Seismic3DObnReceiverGathersTemplate"):
            GridOverrider().run(
                headers,
                index_names=("shot_line", "gun", "shot_point"),
                grid_overrides={"CalculateShotIndex": True},
                template=template,
            )

    def test_shot_wrap_without_template_raises(self) -> None:
        """Omitting the template is the same misconfiguration as passing the wrong type."""
        headers = _make_struct(
            {
                "sail_line": np.array([1], dtype=np.int32),
                "gun": np.array([1], dtype=np.int32),
                "shot_point": np.array([1], dtype=np.int32),
                "cable": np.array([1], dtype=np.int32),
                "channel": np.array([1], dtype=np.int32),
            }
        )
        with pytest.raises(TypeError, match="auto_shot_wrap"):
            GridOverrider().run(
                headers,
                index_names=("sail_line", "gun", "shot_point"),
                grid_overrides={"AutoShotWrap": True},
                template=None,
            )

    def test_header_keys_missing_raises_via_shim(self) -> None:
        """``AutoShotWrap`` + correct template but missing ``cable`` -> :class:`GridOverrideKeysError`.

        Uses the ``StreamerFieldRecords3D`` template (the only template v1.1's
        ``AutoShotWrap`` accepts) so the template-type check passes and we exercise the
        header-key validator instead.
        """
        headers = _make_struct(
            {
                "sail_line": np.array([1, 1], dtype=np.int32),
                "gun": np.array([1, 2], dtype=np.int32),
                "shot_point": np.array([1, 2], dtype=np.int32),
            }
        )
        template = TemplateRegistry().get("StreamerFieldRecords3D")
        with pytest.raises(GridOverrideKeysError, match="ShotWrappingStrategy"):
            GridOverrider().run(
                headers,
                index_names=("sail_line", "gun", "shot_point"),
                grid_overrides={"AutoShotWrap": True},
                template=template,
            )
