"""Environment variable management for MDIO operations."""

from psutil import cpu_count
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


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

    model_config = SettingsConfigDict(case_sensitive=True)

    @field_validator("save_segy_file_header", "raw_headers", "ignore_checks", "cloud_native", mode="before")
    @classmethod
    def parse_bool_fields(cls, v: object) -> bool:
        """Parse boolean fields leniently, like the original implementation."""
        if v is None:
            return False
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "on")
        return bool(v)
