"""This module contains a Pydantic model to parameterize compressors.

Important Objects:
    - Blosc: A Pydantic model that represents a Blosc compression setup.
    - ZFP: Class that represents the ZFP compression model.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field
from pydantic import model_validator
from zarr.codecs import BloscCname
from zarr.codecs import BloscShuffle

from mdio.builder.schemas.core import CamelCaseStrictModel


class Blosc(CamelCaseStrictModel):
    """Data Model for Blosc options."""

    name: str = Field(default="blosc", description="Name of the compressor.")
    cname: BloscCname = Field(default=BloscCname.zstd, description="Compression algorithm name.")
    clevel: int = Field(default=5, ge=0, le=9, description="Compression level (integer 0–9)")
    shuffle: BloscShuffle | None = Field(default=None, description="Shuffling mode before compression.")
    typesize: int | None = Field(default=None, description="The size in bytes that the shuffle is performed over.")
    blocksize: int = Field(default=0, description="The size (in bytes) of blocks to divide data before compression.")


zfp_mode_map = {
    "fixed_rate": 2,
    "fixed_precision": 3,
    "fixed_accuracy": 4,
    "reversible": 5,
}


class ZFPMode(StrEnum):
    """Enum for ZFP algorithm modes."""

    FIXED_RATE = "fixed_rate"
    FIXED_PRECISION = "fixed_precision"
    FIXED_ACCURACY = "fixed_accuracy"
    REVERSIBLE = "reversible"

    @property
    def int_code(self) -> int:
        """Return the integer code of ZFP mode."""
        return zfp_mode_map[self.value]


class ZFP(CamelCaseStrictModel):
    """Data Model for ZFP options."""

    name: str = Field(default="zfp", description="Name of the compressor.")
    mode: ZFPMode = Field()

    tolerance: float | None = Field(
        default=None,
        description="Fixed accuracy in terms of absolute error tolerance.",
    )

    rate: float | None = Field(
        default=None,
        description="Fixed rate in terms of number of compressed bits per value.",
    )

    precision: int | None = Field(
        default=None,
        description="Fixed precision in terms of number of uncompressed bits per value.",
    )

    @model_validator(mode="after")
    def check_requirements(self) -> ZFP:
        """Check if ZFP parameters make sense."""
        mode = self.mode

        # Check if reversible mode is provided without other parameters.
        if mode == ZFPMode.REVERSIBLE and any(
            getattr(self, key) is not None for key in ["tolerance", "rate", "precision"]
        ):
            msg = "Other fields must be None in REVERSIBLE mode"
            raise ValueError(msg)

        if mode == ZFPMode.FIXED_ACCURACY and self.tolerance is None:
            msg = "Tolerance required for FIXED_ACCURACY mode"
            raise ValueError(msg)

        if mode == ZFPMode.FIXED_RATE and self.rate is None:
            msg = "Rate required for FIXED_RATE mode"
            raise ValueError(msg)

        if mode == ZFPMode.FIXED_PRECISION and self.precision is None:
            msg = "Precision required for FIXED_PRECISION mode"
            raise ValueError(msg)

        return self


class CompressorModel(CamelCaseStrictModel):
    """Model representing compressor configuration."""

    compressor: Blosc | ZFP | None = Field(default=None, description="Compression settings.")
