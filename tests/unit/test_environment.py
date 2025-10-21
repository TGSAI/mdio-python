"""Tests for the MDIO Environment API."""

import os
from collections.abc import Callable
from unittest.mock import patch

import pytest

from mdio.api._environ import Environment
from mdio.converters.exceptions import EnvironmentFormatError


class TestEnvironment:
    """Test the Environment API class methods."""

    @pytest.mark.parametrize(
        ("method", "expected_type"),
        [
            (Environment.export_cpus, int),
            (Environment.import_cpus, int),
            (Environment.grid_sparsity_ratio_warn, float),
            (Environment.grid_sparsity_ratio_limit, float),
            (Environment.save_segy_file_header, bool),
            (Environment.raw_headers, bool),
            (Environment.ignore_checks, bool),
            (Environment.cloud_native, bool),
        ],
    )
    def test_default_values(self, method: Callable[[], object], expected_type: type) -> None:
        """Test all methods return correct types with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            result = method()
            assert isinstance(result, expected_type)

    def test_mdio_segy_spec_defaults_to_none(self) -> None:
        """Test mdio_segy_spec returns None by default."""
        with patch.dict(os.environ, {}, clear=True):
            result = Environment.mdio_segy_spec()
            assert result is None

    @pytest.mark.parametrize(
        ("env_var", "value", "method", "expected"),
        [
            ("MDIO__EXPORT__CPU_COUNT", "8", Environment.export_cpus, 8),
            ("MDIO__IMPORT__CPU_COUNT", "4", Environment.import_cpus, 4),
            ("MDIO__GRID__SPARSITY_RATIO_WARN", "3.5", Environment.grid_sparsity_ratio_warn, 3.5),
            ("MDIO__GRID__SPARSITY_RATIO_LIMIT", "15.0", Environment.grid_sparsity_ratio_limit, 15.0),
            ("MDIO__SEGY__SPEC", "/path/to/spec.json", Environment.mdio_segy_spec, "/path/to/spec.json"),
        ],
    )
    def test_env_var_overrides(self, env_var: str, value: str, method: Callable[[], object], expected: object) -> None:
        """Test environment variables override defaults."""
        with patch.dict(os.environ, {env_var: value}):
            result = method()
            assert result == expected

    @pytest.mark.parametrize(
        ("method", "env_var"),
        [
            (Environment.save_segy_file_header, "MDIO__IMPORT__SAVE_SEGY_FILE_HEADER"),
            (Environment.raw_headers, "MDIO__IMPORT__RAW_HEADERS"),
            (Environment.ignore_checks, "MDIO_IGNORE_CHECKS"),
            (Environment.cloud_native, "MDIO__IMPORT__CLOUD_NATIVE"),
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
    def test_boolean_parsing(self, method: Callable[[], bool], env_var: str, value: str, expected: bool) -> None:
        """Test boolean parsing for all boolean methods."""
        with patch.dict(os.environ, {env_var: value}):
            result = method()
            assert result is expected

    @pytest.mark.parametrize(
        ("method", "env_var"),
        [
            (Environment.export_cpus, "MDIO__EXPORT__CPU_COUNT"),
            (Environment.import_cpus, "MDIO__IMPORT__CPU_COUNT"),
        ],
    )
    @pytest.mark.parametrize("invalid_value", ["invalid", "not_a_number", ""])
    def test_int_validation_errors(self, method: Callable[[], int], env_var: str, invalid_value: str) -> None:
        """Test integer methods raise errors for invalid values."""
        with patch.dict(os.environ, {env_var: invalid_value}):
            with pytest.raises(EnvironmentFormatError) as exc_info:
                method()
            assert env_var in str(exc_info.value)
            assert "int" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("method", "env_var"),
        [
            (Environment.grid_sparsity_ratio_warn, "MDIO__GRID__SPARSITY_RATIO_WARN"),
            (Environment.grid_sparsity_ratio_limit, "MDIO__GRID__SPARSITY_RATIO_LIMIT"),
        ],
    )
    @pytest.mark.parametrize("invalid_value", ["invalid", "not_a_number", ""])
    def test_float_validation_errors(self, method: Callable[[], float], env_var: str, invalid_value: str) -> None:
        """Test float methods raise errors for invalid values."""
        with patch.dict(os.environ, {env_var: invalid_value}):
            with pytest.raises(EnvironmentFormatError) as exc_info:
                method()
            assert env_var in str(exc_info.value)
            assert "float" in str(exc_info.value)

    def test_environment_isolation(self) -> None:
        """Test that environment changes don't affect other tests."""
        original_values = {
            "cpus": Environment.export_cpus(),
            "ratio": Environment.grid_sparsity_ratio_warn(),
            "bool": Environment.save_segy_file_header(),
        }

        with patch.dict(
            os.environ,
            {
                "MDIO__EXPORT__CPU_COUNT": "99",
                "MDIO__GRID__SPARSITY_RATIO_WARN": "99.9",
                "MDIO__IMPORT__SAVE_SEGY_FILE_HEADER": "true",
            },
        ):
            assert Environment.export_cpus() == 99
            assert Environment.grid_sparsity_ratio_warn() == 99.9
            assert Environment.save_segy_file_header() is True

        # Values should be restored after context
        assert Environment.export_cpus() == original_values["cpus"]
        assert Environment.grid_sparsity_ratio_warn() == original_values["ratio"]
        assert Environment.save_segy_file_header() == original_values["bool"]
