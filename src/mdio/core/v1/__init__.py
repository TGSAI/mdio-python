"""MDIO core v1 package initialization.

Exposes the MDIO overloads and core v1 functionality.
"""

from ._overloads import mdio
from ._serializer import make_coordinate
from ._serializer import make_dataset
from ._serializer import make_dataset_metadata
from ._serializer import make_named_dimension
from ._serializer import make_variable
from .builder import MDIODatasetBuilder
from .factory import SCHEMA_TEMPLATE_MAP
from .factory import MDIOSchemaType

__all__ = [
    "MDIODatasetBuilder",
    "make_coordinate",
    "make_dataset",
    "make_dataset_metadata",
    "make_named_dimension",
    "make_variable",
    "mdio",
    "MDIOSchemaType",
    "SCHEMA_TEMPLATE_MAP",
]
