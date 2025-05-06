"""MDIO core v1 package initialization.

Exposes the MDIO overloads and core v1 functionality.
"""

from ._overloads import mdio
from .builder import MDIODatasetBuilder
from .builder import write_mdio_metadata
from ._serializer import (
    make_coordinate,
    make_dataset,
    make_dataset_metadata,
    make_named_dimension,
    make_variable,
)
from .factory import AbstractTemplateFactory


__all__ = [
    "MDIODatasetBuilder",
    "AbstractTemplateFactory",
    "make_coordinate",
    "make_dataset",
    "make_dataset_metadata",
    "make_named_dimension",
    "make_variable",
    "mdio",
    "write_mdio_metadata",
]
