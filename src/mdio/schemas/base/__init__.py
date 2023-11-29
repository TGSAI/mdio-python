"""Base schemas to be used downstream."""


from typing import TypeAlias

from mdio.schemas.base.blosc import Blosc
from mdio.schemas.base.zfp import ZFP


Compressors: TypeAlias = Blosc | ZFP


__all__ = [
    "Compressors",
]
