"""Tests for the MDIO Environment API."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

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
            "save_header": MDIOSettings().save_segy_file_header,
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
            assert MDIOSettings().save_segy_file_header == 1

        # Values should be restored after context
        assert MDIOSettings().export_cpus == original_values["cpus"]
        assert MDIOSettings().grid_sparsity_ratio_warn == original_values["ratio"]
        assert MDIOSettings().save_segy_file_header == original_values["save_header"]


class TestSaveSegyFileHeaderMode:
    """Test coercion for ``MDIO__IMPORT__SAVE_SEGY_FILE_HEADER``."""

    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [
            ("0", 0),
            ("1", 1),
            ("2", 2),
            ("false", 0),
            ("False", 0),
            ("FALSE", 0),
            ("no", 0),
            ("off", 0),
            ("true", 1),
            ("True", 1),
            ("TRUE", 1),
            ("yes", 1),
            ("on", 1),
        ],
    )
    def test_string_coercion(self, env_value: str, expected: int) -> None:
        """Strings (including legacy bool aliases) coerce to 0, 1, or 2."""
        with patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": env_value}):
            assert MDIOSettings().save_segy_file_header == expected

    @pytest.mark.parametrize("python_value", [False, True, 0, 1, 2])
    def test_native_python_values(self, python_value: bool | int) -> None:
        """Bool/int passed directly are accepted for backwards compatibility."""
        settings = MDIOSettings(MDIO__IMPORT__SAVE_SEGY_FILE_HEADER=python_value)
        assert settings.save_segy_file_header == int(python_value)

    @pytest.mark.parametrize("bad_value", ["3", "-1", "maybe", "tru"])
    def test_rejects_invalid_strings(self, bad_value: str) -> None:
        """Anything other than 0/1/2 or bool aliases is rejected."""
        with (
            patch.dict(os.environ, {"MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": bad_value}),
            pytest.raises(ValidationError),
        ):
            MDIOSettings()
