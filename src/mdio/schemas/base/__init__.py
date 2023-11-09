"""Base schemas to be used downstream."""


from typing import TypeAlias

from mdio.schemas.base.blosc import Blosc
from mdio.schemas.base.numeric import DataType
from mdio.schemas.base.numeric import StructuredDataType
from mdio.schemas.base.zfp import ZFP


Compressors: TypeAlias = Blosc | ZFP


__all__ = [
    "Compressors",
    "DataType",
    "StructuredDataType",
]
