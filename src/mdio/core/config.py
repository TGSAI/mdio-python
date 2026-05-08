"""Environment variable management for MDIO operations."""

from typing import Literal

from psutil import cpu_count
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

SAVE_SEGY_FILE_HEADER_OFF = 0
SAVE_SEGY_FILE_HEADER_STRICT = 1
SAVE_SEGY_FILE_HEADER_LENIENT = 2

SaveSegyFileHeaderMode = Literal[
    SAVE_SEGY_FILE_HEADER_OFF,
    SAVE_SEGY_FILE_HEADER_STRICT,
    SAVE_SEGY_FILE_HEADER_LENIENT,
]

_SAVE_HEADER_TRUE_STRINGS = frozenset({"true", "yes", "on"})
_SAVE_HEADER_FALSE_STRINGS = frozenset({"false", "no", "off"})


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
    save_segy_file_header: SaveSegyFileHeaderMode = Field(
        default=0,
        description=(
            "How to save SEG-Y file headers: 0 (or False) skips, 1 (or True) saves "
            "and raises on malformed text header, 2 saves and corrects malformed text header."
        ),
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

    @field_validator("save_segy_file_header", mode="before")
    @classmethod
    def _coerce_save_segy_file_header(cls, value: object) -> object:
        """Accept legacy bool values and case-insensitive string aliases."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _SAVE_HEADER_FALSE_STRINGS:
                return SAVE_SEGY_FILE_HEADER_OFF
            if normalized in _SAVE_HEADER_TRUE_STRINGS:
                return SAVE_SEGY_FILE_HEADER_STRICT
            try:
                return int(value)
            except ValueError:
                pass
        if isinstance(value, bool):
            return int(value)
        return value
