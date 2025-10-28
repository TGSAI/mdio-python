"""Tests for the MDIO Environment API."""

import os
from unittest.mock import patch

import pytest

from mdio.core.config import MDIOSettings


class TestEnvironment:
    """Test the Environment API module functions."""

    @pytest.mark.parametrize(
        ("env_var", "value", "property_name", "expected"),
        [
            ("MDIO__EXPORT__CPU_COUNT", "8", "export_cpus", 8),
            ("MDIO__IMPORT__CPU_COUNT", "4", "import_cpus", 4),
            ("MDIO__GRID__SPARSITY_RATIO_WARN", "3.5", "grid_sparsity_ratio_warn", 3.5),
            ("MDIO__GRID__SPARSITY_RATIO_LIMIT", "15.0", "grid_sparsity_ratio_limit", 15.0),
        ],
    )
    def test_env_var_overrides(self, env_var: str, value: str, property_name: str, expected: object) -> None:
        """Test environment variables override defaults."""
        with patch.dict(os.environ, {env_var: value}):
            settings = MDIOSettings()
            result = getattr(settings, property_name)
            assert result == expected

    def test_environment_isolation(self) -> None:
        """Test that environment changes don't affect other tests."""
        original_values = {
            "cpus": MDIOSettings().export_cpus,
            "ratio": MDIOSettings().grid_sparsity_ratio_warn,
            "bool": MDIOSettings().save_segy_file_header,
        }

        with patch.dict(
            os.environ,
            {
                "MDIO__EXPORT__CPU_COUNT": "99",
                "MDIO__GRID__SPARSITY_RATIO_WARN": "99.9",
                "MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "true",
            },
        ):
            assert MDIOSettings().export_cpus == 99
            assert MDIOSettings().grid_sparsity_ratio_warn == 99.9
            assert MDIOSettings().save_segy_file_header is True

        # Values should be restored after context
        assert MDIOSettings().export_cpus == original_values["cpus"]
        assert MDIOSettings().grid_sparsity_ratio_warn == original_values["ratio"]
        assert MDIOSettings().save_segy_file_header == original_values["bool"]
