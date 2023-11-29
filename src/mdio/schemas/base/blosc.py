"""This module contains a Pydantic model to parameterize the Blosc compression.

Classes:
    - BloscAlgorithm: A string enumeration that represents the Blosc
      compression algorithms.
    - BloscShuffle: An integer enumeration that represents the Blosc shuffle
      strategies.
    - Blosc: A Pydantic model that represents a Blosc compression setup.

Features:
    - BloscAlgorithm includes several different Blosc algorithms such as
      "blosclz", "lz4", "lz4hc", "zlib", and "zstd".
    - BloscShuffle includes several different Blosc shuffling strategies,
      including "NOSHUFFLE", "SHUFFLE", "BITSHUFFLE", and "AUTOSHUFFLE".
    - Blosc has various attributes to setup the Blosc compression library
      such as "name", "level", "shuffle", and "blocksize".
    - The "name" attribute must be an instance of BloscAlgorithm.
    - The "level" attribute must be an integer between 0 and 9, inclusive.
    - The "shuffle" attribute must be an instance of BloscShuffle.
    - The "blocksize" attribute must be an integer, the default value is 0.

Note:
    This module requires pydantic, which is used for data validation using
    Python type annotations.
"""


from enum import Enum
from enum import StrEnum

from pydantic import Field

from mdio.schemas.base.core import StrictCamelBaseModel


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


class Blosc(StrictCamelBaseModel):
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
