"""Base schemas to be used downstream."""


from typing import TypeAlias

from mdio.schemas.base.blosc import Blosc
from mdio.schemas.base.dimension import Dimension
from mdio.schemas.base.metadata import UserAttributes
from mdio.schemas.base.scalar import DataType
from mdio.schemas.base.scalar import StructuredDataType
from mdio.schemas.base.zfp import ZFP


Compressors: TypeAlias = Blosc | ZFP


__all__ = [
    "Compressors",
    "Dimension",
    "UserAttributes",
    "DataType",
    "StructuredDataType",
]
