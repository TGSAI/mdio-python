"""Environment variable management for MDIO operations."""

from os import getenv
from typing import Final

from psutil import cpu_count

from mdio.converters.exceptions import EnvironmentFormatError


class Environment:
    """Unified API for accessing and validating MDIO environment variables."""

    # Environment variable keys and defaults
    _EXPORT_CPUS_KEY: Final[str] = "MDIO__EXPORT__CPU_COUNT"
    _IMPORT_CPUS_KEY: Final[str] = "MDIO__IMPORT__CPU_COUNT"
    _GRID_SPARSITY_RATIO_WARN_KEY: Final[str] = "MDIO__GRID__SPARSITY_RATIO_WARN"
    _GRID_SPARSITY_RATIO_LIMIT_KEY: Final[str] = "MDIO__GRID__SPARSITY_RATIO_LIMIT"
    _SAVE_SEGY_FILE_HEADER_KEY: Final[str] = "MDIO__IMPORT__SAVE_SEGY_FILE_HEADER"
    _MDIO_SEGY_SPEC_KEY: Final[str] = "MDIO__SEGY__SPEC"
    _RAW_HEADERS_KEY: Final[str] = "MDIO__IMPORT__RAW_HEADERS"
    _IGNORE_CHECKS_KEY: Final[str] = "MDIO_IGNORE_CHECKS"
    _CLOUD_NATIVE_KEY: Final[str] = "MDIO__IMPORT__CLOUD_NATIVE"

    # Default values
    _EXPORT_CPUS_DEFAULT: Final[int] = cpu_count(logical=True)
    _IMPORT_CPUS_DEFAULT: Final[int] = cpu_count(logical=True)
    _GRID_SPARSITY_RATIO_WARN_DEFAULT: Final[str] = "2"
    _GRID_SPARSITY_RATIO_LIMIT_DEFAULT: Final[str] = "10"
    _SAVE_SEGY_FILE_HEADER_DEFAULT: Final[str] = "false"
    _MDIO_SEGY_SPEC_DEFAULT: Final[None] = None
    _RAW_HEADERS_DEFAULT: Final[str] = "false"
    _IGNORE_CHECKS_DEFAULT: Final[str] = "false"
    _CLOUD_NATIVE_DEFAULT: Final[str] = "false"

    @classmethod
    def _get_env_value(cls, key: str, default: str | int | None) -> str | None:
        """Get environment variable value with fallback to default."""
        if isinstance(default, int):
            default = str(default)
        return getenv(key, default)

    @staticmethod
    def _parse_bool(value: str | None) -> bool:
        """Parse string value to boolean."""
        if value is None:
            return False
        return value.lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _parse_int(value: str | None, key: str) -> int:
        """Parse string value to integer with validation."""
        if value is None:
            raise EnvironmentFormatError(key, "int")
        try:
            return int(value)
        except ValueError as e:
            raise EnvironmentFormatError(key, "int") from e

    @staticmethod
    def _parse_float(value: str | None, key: str) -> float:
        """Parse string value to float with validation."""
        if value is None:
            raise EnvironmentFormatError(key, "float")
        try:
            return float(value)
        except ValueError as e:
            raise EnvironmentFormatError(key, "float") from e

    @classmethod
    def export_cpus(cls) -> int:
        """Number of CPUs to use for export operations."""
        value = cls._get_env_value(cls._EXPORT_CPUS_KEY, cls._EXPORT_CPUS_DEFAULT)
        return cls._parse_int(value, cls._EXPORT_CPUS_KEY)

    @classmethod
    def import_cpus(cls) -> int:
        """Number of CPUs to use for import operations."""
        value = cls._get_env_value(cls._IMPORT_CPUS_KEY, cls._IMPORT_CPUS_DEFAULT)
        return cls._parse_int(value, cls._IMPORT_CPUS_KEY)

    @classmethod
    def grid_sparsity_ratio_warn(cls) -> float:
        """Sparsity ratio threshold for warnings."""
        value = cls._get_env_value(cls._GRID_SPARSITY_RATIO_WARN_KEY, cls._GRID_SPARSITY_RATIO_WARN_DEFAULT)
        return cls._parse_float(value, cls._GRID_SPARSITY_RATIO_WARN_KEY)

    @classmethod
    def grid_sparsity_ratio_limit(cls) -> float:
        """Sparsity ratio threshold for errors."""
        value = cls._get_env_value(cls._GRID_SPARSITY_RATIO_LIMIT_KEY, cls._GRID_SPARSITY_RATIO_LIMIT_DEFAULT)
        return cls._parse_float(value, cls._GRID_SPARSITY_RATIO_LIMIT_KEY)

    @classmethod
    def save_segy_file_header(cls) -> bool:
        """Whether to save SEG-Y file headers."""
        value = cls._get_env_value(cls._SAVE_SEGY_FILE_HEADER_KEY, cls._SAVE_SEGY_FILE_HEADER_DEFAULT)
        return cls._parse_bool(value)

    @classmethod
    def mdio_segy_spec(cls) -> str | None:
        """Path to MDIO SEG-Y specification file."""
        return cls._get_env_value(cls._MDIO_SEGY_SPEC_KEY, cls._MDIO_SEGY_SPEC_DEFAULT)

    @classmethod
    def raw_headers(cls) -> bool:
        """Whether to preserve raw headers."""
        value = cls._get_env_value(cls._RAW_HEADERS_KEY, cls._RAW_HEADERS_DEFAULT)
        return cls._parse_bool(value)

    @classmethod
    def ignore_checks(cls) -> bool:
        """Whether to ignore validation checks."""
        value = cls._get_env_value(cls._IGNORE_CHECKS_KEY, cls._IGNORE_CHECKS_DEFAULT)
        return cls._parse_bool(value)

    @classmethod
    def cloud_native(cls) -> bool:
        """Whether to use cloud-native mode for SEG-Y processing."""
        value = cls._get_env_value(cls._CLOUD_NATIVE_KEY, cls._CLOUD_NATIVE_DEFAULT)
        return cls._parse_bool(value)
