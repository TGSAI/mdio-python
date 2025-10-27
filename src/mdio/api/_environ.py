"""Environment variable management for MDIO operations."""

from psutil import cpu_count
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator
from pydantic_settings import BaseSettings

from mdio.converters.exceptions import EnvironmentFormatError


class MDIOSettings(BaseSettings):
    """MDIO environment configuration settings."""

    # CPU configuration
    export_cpus: int = Field(
        default_factory=lambda: cpu_count(logical=True),
        description="Number of CPUs to use for export operations",
        alias="MDIO__EXPORT__CPU_COUNT",
    )
    import_cpus: int = Field(
        default_factory=lambda: cpu_count(logical=True),
        description="Number of CPUs to use for import operations",
        alias="MDIO__IMPORT__CPU_COUNT",
    )

    # Grid sparsity configuration
    grid_sparsity_ratio_warn: float = Field(
        default=2.0,
        description="Sparsity ratio threshold for warnings",
        alias="MDIO__GRID__SPARSITY_RATIO_WARN",
    )
    grid_sparsity_ratio_limit: float = Field(
        default=10.0,
        description="Sparsity ratio threshold for errors",
        alias="MDIO__GRID__SPARSITY_RATIO_LIMIT",
    )

    # Import configuration
    save_segy_file_header: bool = Field(
        default=False,
        description="Whether to save SEG-Y file headers",
        alias="MDIO__IMPORT__SAVE_SEGY_FILE_HEADER",
    )
    raw_headers: bool = Field(
        default=False,
        description="Whether to preserve raw headers",
        alias="MDIO__IMPORT__RAW_HEADERS",
    )
    cloud_native: bool = Field(
        default=False,
        description="Whether to use cloud-native mode for SEG-Y processing",
        alias="MDIO__IMPORT__CLOUD_NATIVE",
    )

    # General configuration
    ignore_checks: bool = Field(
        default=False,
        description="Whether to ignore validation checks",
        alias="MDIO_IGNORE_CHECKS",
    )

    model_config = ConfigDict(
        env_prefix="",
        case_sensitive=True,
    )

    @field_validator("save_segy_file_header", "raw_headers", "ignore_checks", "cloud_native", mode="before")
    @classmethod
    def parse_bool_fields(cls, v: object) -> bool:
        """Parse boolean fields leniently, like the original implementation."""
        if v is None:
            return False
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "on")
        return bool(v)


def _get_settings() -> MDIOSettings:
    """Get current MDIO settings from environment variables."""
    try:
        return MDIOSettings()
    except ValidationError as e:
        # Extract the field name and expected type from the error
        error_details = e.errors()[0]
        field_name = error_details.get("loc", [None])[0]
        error_type = error_details.get("type", "unknown")

        # Map pydantic error types to our error types
        type_mapping = {
            "int_parsing": "int",
            "float_parsing": "float",
        }
        mapped_type = type_mapping.get(error_type, error_type)

        # Map field names back to environment variable names for the error
        env_var_mapping = {
            "export_cpus": "MDIO__EXPORT__CPU_COUNT",
            "import_cpus": "MDIO__IMPORT__CPU_COUNT",
            "grid_sparsity_ratio_warn": "MDIO__GRID__SPARSITY_RATIO_WARN",
            "grid_sparsity_ratio_limit": "MDIO__GRID__SPARSITY_RATIO_LIMIT",
        }
        env_var = env_var_mapping.get(field_name, field_name)

        raise EnvironmentFormatError(env_var, mapped_type) from e


def export_cpus() -> int:
    """Number of CPUs to use for export operations."""
    return _get_settings().export_cpus


def import_cpus() -> int:
    """Number of CPUs to use for import operations."""
    return _get_settings().import_cpus


def grid_sparsity_ratio_warn() -> float:
    """Sparsity ratio threshold for warnings."""
    return _get_settings().grid_sparsity_ratio_warn


def grid_sparsity_ratio_limit() -> float:
    """Sparsity ratio threshold for errors."""
    return _get_settings().grid_sparsity_ratio_limit


def save_segy_file_header() -> bool:
    """Whether to save SEG-Y file headers."""
    return _get_settings().save_segy_file_header


def raw_headers() -> bool:
    """Whether to preserve raw headers."""
    return _get_settings().raw_headers


def ignore_checks() -> bool:
    """Whether to ignore validation checks."""
    return _get_settings().ignore_checks


def cloud_native() -> bool:
    """Whether to use cloud-native mode for SEG-Y processing."""
    return _get_settings().cloud_native
