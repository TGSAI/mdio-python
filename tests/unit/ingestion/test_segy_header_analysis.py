"""Tests for SEG-Y header analysis acquisition-geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mdio.ingestion.segy.header_analysis import ShotGunGeometryType
from mdio.ingestion.segy.header_analysis import StreamerShotGeometryType
from mdio.ingestion.segy.header_analysis import analyze_lines_for_guns
from mdio.ingestion.segy.header_analysis import analyze_non_indexed_headers
from mdio.ingestion.segy.header_analysis import analyze_saillines_for_guns
from mdio.ingestion.segy.header_analysis import analyze_streamer_headers
from mdio.ingestion.segy.header_analysis import create_counter
from mdio.ingestion.segy.header_analysis import create_trace_index
from tests.unit.ingestion.testing_helpers import make_header_array


def _streamer_headers(records: list[tuple[int, int]]) -> np.ndarray:
    """Build a (cable, channel) header array from ``(cable, channel)`` pairs."""
    cables, channels = zip(*records, strict=True)
    return make_header_array(
        {
            "cable": np.asarray(cables, dtype=np.int32),
            "channel": np.asarray(channels, dtype=np.int32),
        }
    )


def _gun_headers(records: list[tuple[int, int, int]], line_field: str = "sail_line") -> np.ndarray:
    """Build a (line_field, shot_point, gun) header array from triples."""
    lines, shots, guns = zip(*records, strict=True)
    return make_header_array(
        {
            line_field: np.asarray(lines, dtype=np.int32),
            "shot_point": np.asarray(shots, dtype=np.int32),
            "gun": np.asarray(guns, dtype=np.int8),
        }
    )


class TestAnalyzeStreamerHeaders:
    """Tests for ``analyze_streamer_headers``."""

    def test_non_overlapping_channels_returns_type_b(self) -> None:
        """Non-overlapping cable channel ranges should produce Configuration B."""
        records: list[tuple[int, int]] = []
        for cable in (1, 2, 3):
            for chan in range(1, 6):
                records.append((cable, (cable - 1) * 5 + chan))

        headers = _streamer_headers(records)

        unique_cables, mins, maxs, geom = analyze_streamer_headers(headers)

        np.testing.assert_array_equal(unique_cables, [1, 2, 3])
        np.testing.assert_array_equal(mins, [1, 6, 11])
        np.testing.assert_array_equal(maxs, [5, 10, 15])
        assert geom is StreamerShotGeometryType.B

    def test_overlapping_channels_returns_type_a(self) -> None:
        """Overlapping channel ranges between cables should produce Configuration A."""
        records: list[tuple[int, int]] = []
        for cable in (1, 2):
            for chan in range(1, 6):
                records.append((cable, chan))

        headers = _streamer_headers(records)
        unique_cables, _, _, geom = analyze_streamer_headers(headers)

        np.testing.assert_array_equal(unique_cables, [1, 2])
        assert geom is StreamerShotGeometryType.A

    def test_single_cable_returns_type_b(self) -> None:
        """A single cable has no neighbours to overlap with → Configuration B."""
        records = [(7, chan) for chan in range(1, 6)]
        headers = _streamer_headers(records)

        unique_cables, mins, maxs, geom = analyze_streamer_headers(headers)

        np.testing.assert_array_equal(unique_cables, [7])
        np.testing.assert_array_equal(mins, [1])
        np.testing.assert_array_equal(maxs, [5])
        assert geom is StreamerShotGeometryType.B


class TestAnalyzeLinesForGuns:
    """Tests for ``analyze_lines_for_guns``."""

    def test_dense_shots_per_gun_returns_type_a(self) -> None:
        """Each gun densely numbered 1..N (overlap across guns) -> Configuration A."""
        # Gun 1: shots 1..4, Gun 2: shots 1..4 (line value 100)
        records = [(100, shot, gun) for gun in (1, 2) for shot in range(1, 5)]
        headers = _gun_headers(records)

        unique_lines, per_line, geom = analyze_lines_for_guns(headers, line_field="sail_line")

        np.testing.assert_array_equal(unique_lines, [100])
        assert per_line == {"100": [1, 2]}
        assert geom is ShotGunGeometryType.A

    def test_interleaved_shots_returns_type_b(self) -> None:
        """Interleaved shot numbering (unique per line, sparse per gun) -> Configuration B."""
        # Gun 1: odd shots, gun 2: even shots, all unique within the same line.
        records = []
        for shot in (1, 3, 5):
            records.append((200, shot, 1))
        for shot in (2, 4, 6):
            records.append((200, shot, 2))
        headers = _gun_headers(records)

        unique_lines, per_line, geom = analyze_lines_for_guns(headers, line_field="sail_line")

        np.testing.assert_array_equal(unique_lines, [200])
        assert per_line == {"200": [1, 2]}
        assert geom is ShotGunGeometryType.B

    def test_custom_line_field(self) -> None:
        """Function must work for any line-field name (e.g. ``shot_line``)."""
        records = [(7, shot, gun) for gun in (1, 2) for shot in range(1, 4)]
        headers = _gun_headers(records, line_field="shot_line")

        unique_lines, per_line, geom = analyze_lines_for_guns(headers, line_field="shot_line")

        np.testing.assert_array_equal(unique_lines, [7])
        assert per_line == {"7": [1, 2]}
        # Dense 1..N per gun → configuration A
        assert geom is ShotGunGeometryType.A

    def test_single_gun_line_stays_type_b(self) -> None:
        """A line with a single gun has nothing to interleave → Configuration B.

        With ``num_guns == 1``, dividing shot points by 1 is the identity, so the
        floor/unique check trivially matches and the function should stay in B.
        """
        records = [(300, shot, 1) for shot in range(1, 5)]
        headers = _gun_headers(records)

        unique_lines, per_line, geom = analyze_lines_for_guns(headers, line_field="sail_line")

        np.testing.assert_array_equal(unique_lines, [300])
        assert per_line == {"300": [1]}
        assert geom is ShotGunGeometryType.B


class TestAnalyzeSailLinesForGuns:
    """Tests for ``analyze_saillines_for_guns``."""

    def test_delegates_to_generic_function(self) -> None:
        """The sail-line variant should mirror ``analyze_lines_for_guns`` with ``sail_line``."""
        records = [(11, shot, gun) for gun in (1, 2) for shot in range(1, 4)]
        headers = _gun_headers(records, line_field="sail_line")

        lines_a, per_line_a, geom_a = analyze_saillines_for_guns(headers)
        lines_b, per_line_b, geom_b = analyze_lines_for_guns(headers, line_field="sail_line")

        np.testing.assert_array_equal(lines_a, lines_b)
        assert per_line_a == per_line_b
        assert geom_a is geom_b
        # Sanity check on the underlying geometry (dense per gun -> A).
        assert geom_a is ShotGunGeometryType.A


class TestCreateCounter:
    """Tests for ``create_counter``."""

    def test_returns_zero_at_max_depth(self) -> None:
        """Reaching the requested total depth yields the leaf integer 0."""
        assert create_counter(2, 2, {}, []) == 0

    def test_builds_nested_dict_per_header(self) -> None:
        """Two-level tree should have a leaf 0 under each combination."""
        unique = {"cable": np.array([1, 2]), "channel": np.array([10, 11, 12])}
        names = ["cable", "channel"]

        tree = create_counter(0, 2, unique, names)

        assert set(tree.keys()) == {1, 2}
        for cable in (1, 2):
            assert set(tree[cable].keys()) == {10, 11, 12}
            for chan in (10, 11, 12):
                assert tree[cable][chan] == 0


class TestCreateTraceIndex:
    """Tests for ``create_trace_index``."""

    def test_returns_none_when_depth_zero(self) -> None:
        """A zero-depth tree means no header names → None."""
        headers = np.array([], dtype=[("cable", "i4")])
        assert create_trace_index(0, {}, headers, []) is None

    def test_assigns_dense_trace_index_for_single_header(self) -> None:
        """One-dim counter assigns 1..N within each unique header value."""
        cable = np.array([1, 1, 2, 2, 2], dtype=np.int32)
        headers = np.empty(cable.size, dtype=[("cable", "i4")])
        headers["cable"] = cable
        counter = {1: 0, 2: 0}

        out = create_trace_index(1, counter, headers, ["cable"])

        assert out is not None
        assert "trace" in out.dtype.names
        np.testing.assert_array_equal(out["trace"], [1, 2, 1, 2, 3])

    def test_assigns_dense_trace_index_for_two_headers(self) -> None:
        """Two-dim counter assigns 1..N within each unique (cable, channel)."""
        cable = np.array([1, 1, 2, 2, 2], dtype=np.int32)
        channel = np.array([10, 10, 20, 20, 30], dtype=np.int32)
        headers = np.empty(cable.size, dtype=[("cable", "i4"), ("channel", "i4")])
        headers["cable"] = cable
        headers["channel"] = channel
        counter = {1: {10: 0, 20: 0, 30: 0}, 2: {10: 0, 20: 0, 30: 0}}

        out = create_trace_index(2, counter, headers, ["cable", "channel"])

        np.testing.assert_array_equal(out["trace"], [1, 2, 1, 2, 1])


class TestAnalyzeNonIndexedHeaders:
    """Tests for ``analyze_non_indexed_headers``."""

    def test_adds_trace_field_with_dense_index(self) -> None:
        """The returned header array should carry a 'trace' field counting within unique keys."""
        cable = np.array([1, 1, 1, 2, 2], dtype=np.int32)
        headers = np.empty(cable.size, dtype=[("cable", "i4")])
        headers["cable"] = cable

        out = analyze_non_indexed_headers(headers)

        assert "trace" in out.dtype.names
        np.testing.assert_array_equal(out["trace"], [1, 2, 3, 1, 2])

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
    def test_respects_dtype_kwarg(self, dtype: type[np.integer]) -> None:
        """The dtype kwarg should drive the 'trace' field's numpy dtype."""
        cable = np.array([1, 2], dtype=np.int32)
        headers = np.empty(cable.size, dtype=[("cable", "i4")])
        headers["cable"] = cable

        out = analyze_non_indexed_headers(headers, dtype=dtype)
        assert out["trace"].dtype == np.dtype(dtype)
