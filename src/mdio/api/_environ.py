"""Environment variable management for MDIO operations."""

from os import getenv

from psutil import cpu_count

from mdio.converters.exceptions import EnvironmentFormatError

# Environment variable keys
_EXPORT_CPUS_KEY = "MDIO__EXPORT__CPU_COUNT"
_IMPORT_CPUS_KEY = "MDIO__IMPORT__CPU_COUNT"
_GRID_SPARSITY_RATIO_WARN_KEY = "MDIO__GRID__SPARSITY_RATIO_WARN"
_GRID_SPARSITY_RATIO_LIMIT_KEY = "MDIO__GRID__SPARSITY_RATIO_LIMIT"
_SAVE_SEGY_FILE_HEADER_KEY = "MDIO__IMPORT__SAVE_SEGY_FILE_HEADER"
_MDIO_SEGY_SPEC_KEY = "MDIO__SEGY__SPEC"
_RAW_HEADERS_KEY = "MDIO__IMPORT__RAW_HEADERS"
_IGNORE_CHECKS_KEY = "MDIO_IGNORE_CHECKS"
_CLOUD_NATIVE_KEY = "MDIO__IMPORT__CLOUD_NATIVE"

# Default values
_EXPORT_CPUS_DEFAULT = cpu_count(logical=True)
_IMPORT_CPUS_DEFAULT = cpu_count(logical=True)
_GRID_SPARSITY_RATIO_WARN_DEFAULT = "2"
_GRID_SPARSITY_RATIO_LIMIT_DEFAULT = "10"
_SAVE_SEGY_FILE_HEADER_DEFAULT = "false"
_MDIO_SEGY_SPEC_DEFAULT = None
_RAW_HEADERS_DEFAULT = "false"
_IGNORE_CHECKS_DEFAULT = "false"
_CLOUD_NATIVE_DEFAULT = "false"


def _get_env_value(key: str, default: str | int | None) -> str | None:
    """Get environment variable value with fallback to default."""
    if isinstance(default, int):
        default = str(default)
    return getenv(key, default)


def _parse_bool(value: str | None) -> bool:
    """Parse string value to boolean."""
    if value is None:
        return False
    return value.lower() in ("1", "true", "yes", "on")


def _parse_int(value: str | None, key: str) -> int:
    """Parse string value to integer with validation."""
    if value is None:
        raise EnvironmentFormatError(key, "int")
    try:
        return int(value)
    except ValueError as e:
        raise EnvironmentFormatError(key, "int") from e


def _parse_float(value: str | None, key: str) -> float:
    """Parse string value to float with validation."""
    if value is None:
        raise EnvironmentFormatError(key, "float")
    try:
        return float(value)
    except ValueError as e:
        raise EnvironmentFormatError(key, "float") from e


def export_cpus() -> int:
    """Number of CPUs to use for export operations."""
    value = _get_env_value(_EXPORT_CPUS_KEY, _EXPORT_CPUS_DEFAULT)
    return _parse_int(value, _EXPORT_CPUS_KEY)


def import_cpus() -> int:
    """Number of CPUs to use for import operations."""
    value = _get_env_value(_IMPORT_CPUS_KEY, _IMPORT_CPUS_DEFAULT)
    return _parse_int(value, _IMPORT_CPUS_KEY)


def grid_sparsity_ratio_warn() -> float:
    """Sparsity ratio threshold for warnings."""
    value = _get_env_value(_GRID_SPARSITY_RATIO_WARN_KEY, _GRID_SPARSITY_RATIO_WARN_DEFAULT)
    return _parse_float(value, _GRID_SPARSITY_RATIO_WARN_KEY)


def grid_sparsity_ratio_limit() -> float:
    """Sparsity ratio threshold for errors."""
    value = _get_env_value(_GRID_SPARSITY_RATIO_LIMIT_KEY, _GRID_SPARSITY_RATIO_LIMIT_DEFAULT)
    return _parse_float(value, _GRID_SPARSITY_RATIO_LIMIT_KEY)


def save_segy_file_header() -> bool:
    """Whether to save SEG-Y file headers."""
    value = _get_env_value(_SAVE_SEGY_FILE_HEADER_KEY, _SAVE_SEGY_FILE_HEADER_DEFAULT)
    return _parse_bool(value)


def mdio_segy_spec() -> str | None:
    """Path to MDIO SEG-Y specification file."""
    return _get_env_value(_MDIO_SEGY_SPEC_KEY, _MDIO_SEGY_SPEC_DEFAULT)


def raw_headers() -> bool:
    """Whether to preserve raw headers."""
    value = _get_env_value(_RAW_HEADERS_KEY, _RAW_HEADERS_DEFAULT)
    return _parse_bool(value)


def ignore_checks() -> bool:
    """Whether to ignore validation checks."""
    value = _get_env_value(_IGNORE_CHECKS_KEY, _IGNORE_CHECKS_DEFAULT)
    return _parse_bool(value)


def cloud_native() -> bool:
    """Whether to use cloud-native mode for SEG-Y processing."""
    value = _get_env_value(_CLOUD_NATIVE_KEY, _CLOUD_NATIVE_DEFAULT)
    return _parse_bool(value)
