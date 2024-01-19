"""This module contains a Pydantic model to parameterize compressors.

Important Objects:
    - Blosc: A Pydantic model that represents a Blosc compression setup.
    - ZFP: Class that represents the ZFP compression model.
"""


from enum import Enum
from enum import StrEnum

from pydantic import Field
from pydantic import model_validator

from mdio.schemas.core import CamelCaseStrictModel


class BloscAlgorithm(StrEnum):
    """Enum for Blosc algorithm options."""

    BLOSCLZ = "blosclz"
    LZ4 = "lz4"
    LZ4HC = "lz4hc"
    ZLIB = "zlib"
    ZSTD = "zstd"


class BloscShuffle(Enum):
    """Enum for Blosc shuffle options."""

    NOSHUFFLE = 0
    SHUFFLE = 1
    BITSHUFFLE = 2
    AUTOSHUFFLE = -1


class Blosc(CamelCaseStrictModel):
    """Data Model for Blosc options."""

    name: str = Field(default="blosc", description="Name of the compressor.")
    algorithm: BloscAlgorithm = Field(
        default=BloscAlgorithm.LZ4,
        description="The Blosc compression algorithm to be used.",
    )
    level: int = Field(default=5, ge=0, le=9, description="The compression level.")
    shuffle: BloscShuffle = Field(
        default=BloscShuffle.SHUFFLE,
        description="The shuffle strategy to be applied before compression.",
    )
    blocksize: int = Field(
        default=0,
        description="The size of the block to be used for compression.",
    )


class ZFPMode(StrEnum):
    """Enum for ZFP algorithm modes."""

    FIXED_ACCURACY = "fixed_accuracy"
    FIXED_PRECISION = "fixed_precision"
    FIXED_RATE = "fixed_rate"
    REVERSIBLE = "reversible"


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

    write_header: bool = Field(
        default=True,
        description="Encode array shape, scalar type, and compression parameters.",
    )

    @model_validator(mode="after")
    def check_requirements(self) -> "ZFP":
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

    compressor: Blosc | ZFP | None = Field(
        default=None, description="Compression settings."
    )
