"""Tests for the MDIO Environment API."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from mdio.core.settings import MDIOSettings


class TestEnvironment:
    """Test the Environment API module functions."""

    @pytest.mark.parametrize(
        ("value", "expected_type"),
        [
            (MDIOSettings().export_cpus, int),
            (MDIOSettings().import_cpus, int),
            (MDIOSettings().grid_sparsity_ratio_warn, float),
            (MDIOSettings().grid_sparsity_ratio_limit, float),
            (MDIOSettings().save_segy_file_header, bool),
            (MDIOSettings().raw_headers, bool),
            (MDIOSettings().ignore_checks, bool),
            (MDIOSettings().cloud_native, bool),
        ],
    )
    def test_default_values(self, value: object, expected_type: type) -> None:
        """Test all properties return correct types with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            assert isinstance(value, expected_type)

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

    @pytest.mark.parametrize(
        ("property_name", "env_var"),
        [
            ("save_segy_file_header", "MDIO__IMPORT__SAVE_SEGY_FILE_HEADER"),
            ("raw_headers", "MDIO__IMPORT__RAW_HEADERS"),
            ("ignore_checks", "MDIO_IGNORE_CHECKS"),
            ("cloud_native", "MDIO__IMPORT__CLOUD_NATIVE"),
        ],
    )
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("1", True),
            ("true", True),
            ("yes", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("off", False),
            ("anything_else", False),
            ("", False),
        ],
    )
    def test_boolean_parsing(self, property_name: str, env_var: str, value: str, expected: bool) -> None:
        """Test boolean parsing for all boolean properties."""
        with patch.dict(os.environ, {env_var: value}):
            settings = MDIOSettings()
            result = getattr(settings, property_name)
            assert result is expected

    @pytest.mark.parametrize(
        ("env_var"),
        [
            ("MDIO__EXPORT__CPU_COUNT"),
            ("MDIO__IMPORT__CPU_COUNT"),
        ],
    )
    @pytest.mark.parametrize("invalid_value", ["invalid", "not_a_number", ""])
    def test_int_validation_errors(self, env_var: str, invalid_value: str) -> None:
        """Test integer properties raise errors for invalid values."""
        with patch.dict(os.environ, {env_var: invalid_value}):
            with pytest.raises(ValidationError) as exc_info:
                MDIOSettings()
            assert env_var in str(exc_info.value)
            assert "int" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("env_var"),
        [
            ("MDIO__GRID__SPARSITY_RATIO_WARN"),
            ("MDIO__GRID__SPARSITY_RATIO_LIMIT"),
        ],
    )
    @pytest.mark.parametrize("invalid_value", ["invalid", "not_a_number", ""])
    def test_float_validation_errors(self, env_var: str, invalid_value: str) -> None:
        """Test float properties raise errors for invalid values."""
        with patch.dict(os.environ, {env_var: invalid_value}):
            with pytest.raises(ValidationError) as exc_info:
                MDIOSettings()
            assert env_var in str(exc_info.value)
            assert "float" in str(exc_info.value)

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
