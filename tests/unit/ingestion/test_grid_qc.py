"""Tests for ingestion grid density / sparsity quality control."""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import numpy as np
import pytest

from mdio.converters.exceptions import GridTraceSparsityError
from mdio.core.dimension import Dimension
from mdio.core.grid import Grid
from mdio.ingestion.grid_qc import grid_density_qc


def _make_grid(shape: tuple[int, ...]) -> Grid:
    """Build a Grid with named dimensions of the given size."""
    names = [f"dim_{idx}" for idx in range(len(shape) - 1)] + ["sample"]
    dims = [Dimension(coords=np.arange(size, dtype=np.int32), name=name) for name, size in zip(names, shape, strict=True)]
    return Grid(dims=dims)


class TestGridDensityQc:
    """Test cases for ``grid_density_qc``."""

    def test_no_warning_when_dense(self, caplog: pytest.LogCaptureFixture) -> None:
        """Dense grids (ratio <= warn) should not log or raise."""
        grid = _make_grid((10, 10, 100))  # 100 grid traces
        with caplog.at_level(logging.WARNING):
            grid_density_qc(grid, num_traces=100)
        assert caplog.records == []

    def test_warns_when_above_warn_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        """Sparsity above warn but below limit logs a warning, no raise."""
        grid = _make_grid((10, 10, 100))  # 100 grid traces
        # warn = 2, error = 10 (defaults); ratio of 5 sits between
        with caplog.at_level(logging.WARNING, logger="mdio.ingestion.grid_qc"):
            grid_density_qc(grid, num_traces=20)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "Sparsity ratio: 5.00" in warnings[0].message

    def test_warning_message_includes_dim_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        """The warning body must include shape, trace counts, and a min/max line per dim.

        Pins the format so refactors that touch ``Grid.get_min`` / ``get_max`` or the
        message template don't silently regress operator-facing output.
        """
        grid = _make_grid((10, 10, 100))  # 100 grid traces
        with caplog.at_level(logging.WARNING, logger="mdio.ingestion.grid_qc"):
            grid_density_qc(grid, num_traces=20)

        message = caplog.records[0].message
        assert "SEG-Y trace count: 20" in message
        assert "grid trace count: 100" in message
        assert "{'dim_0': 10, 'dim_1': 10, 'sample': 100}" in message
        for dim_name in ("dim_0", "dim_1", "sample"):
            assert f"\n{dim_name} min: 0 max:" in message

    def test_raises_when_above_limit(self) -> None:
        """Sparsity above the error limit should raise."""
        grid = _make_grid((10, 10, 100))  # 100 grid traces, limit default = 10
        with pytest.raises(GridTraceSparsityError):
            grid_density_qc(grid, num_traces=5)  # ratio 20 > 10

    def test_ignore_checks_suppresses_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Setting MDIO_IGNORE_CHECKS still warns but never raises."""
        grid = _make_grid((10, 10, 100))
        with patch.dict(os.environ, {"MDIO_IGNORE_CHECKS": "1"}), caplog.at_level(
            logging.WARNING, logger="mdio.ingestion.grid_qc"
        ):
            grid_density_qc(grid, num_traces=5)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    def test_zero_traces_treated_as_infinite_sparsity(self) -> None:
        """A SEG-Y with zero traces should be flagged via the limit branch."""
        grid = _make_grid((2, 2, 5))
        with pytest.raises(GridTraceSparsityError):
            grid_density_qc(grid, num_traces=0)

    @pytest.mark.parametrize(
        ("warn", "limit", "num_traces", "expect_raise", "expect_warn"),
        [
            ("100", "1000", "100", False, False),  # ratio 1, both safe
            ("0.5", "1000", "100", False, True),  # ratio 1 > 0.5 (warn only)
            ("0.5", "0.9", "100", True, True),  # ratio 1 > 0.9 (raise)
        ],
    )
    def test_thresholds_respect_env_vars(  # noqa: PLR0913
        self,
        warn: str,
        limit: str,
        num_traces: str,
        expect_raise: bool,
        expect_warn: bool,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Custom warn/limit env vars should drive the QC behavior."""
        grid = _make_grid((10, 10, 100))  # 100 grid traces
        env = {
            "MDIO__GRID__SPARSITY_RATIO_WARN": warn,
            "MDIO__GRID__SPARSITY_RATIO_LIMIT": limit,
        }
        with patch.dict(os.environ, env), caplog.at_level(logging.WARNING, logger="mdio.ingestion.grid_qc"):
            if expect_raise:
                with pytest.raises(GridTraceSparsityError):
                    grid_density_qc(grid, num_traces=int(num_traces))
            else:
                grid_density_qc(grid, num_traces=int(num_traces))

        warned = any(r.levelno == logging.WARNING for r in caplog.records)
        assert warned == expect_warn
