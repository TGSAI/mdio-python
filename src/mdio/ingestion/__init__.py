"""MDIO Ingestion Pipeline Components.

This package contains the ingestion pipeline with clear separation of concerns:
- Schema resolution
- Header analysis
- Index strategies
- Dataset building
- Coordinate utilities
- Validation
- Metadata handling
"""

from mdio.ingestion.coordinate_utils import get_coordinates
from mdio.ingestion.coordinate_utils import get_spatial_coordinate_unit
from mdio.ingestion.coordinate_utils import populate_coordinates
from mdio.ingestion.coordinate_utils import update_template_units
from mdio.ingestion.dataset_factory import DatasetFactory
from mdio.ingestion.header_analysis import ShotGunGeometryType
from mdio.ingestion.header_analysis import StreamerShotGeometryType
from mdio.ingestion.header_analyzer import HeaderAnalyzer
from mdio.ingestion.header_analyzer import HeaderRequirements
from mdio.ingestion.index_strategies import ChannelWrappingStrategy
from mdio.ingestion.index_strategies import CompositeStrategy
from mdio.ingestion.index_strategies import DuplicateHandlingStrategy
from mdio.ingestion.index_strategies import IndexStrategy
from mdio.ingestion.index_strategies import IndexStrategyFactory
from mdio.ingestion.index_strategies import NonBinnedStrategy
from mdio.ingestion.index_strategies import RegularGridStrategy
from mdio.ingestion.index_strategies import ShotWrappingStrategy
from mdio.ingestion.metadata import add_grid_override_to_metadata
from mdio.ingestion.metadata import add_segy_file_headers
from mdio.ingestion.pipeline import run_segy_ingestion
from mdio.ingestion.schema_resolver import CoordinateSpec
from mdio.ingestion.schema_resolver import DimensionSpec
from mdio.ingestion.schema_resolver import ResolvedSchema
from mdio.ingestion.schema_resolver import SchemaResolver
from mdio.ingestion.validation import grid_density_qc
from mdio.ingestion.validation import validate_spec_in_template

__all__ = [
    # Schema resolution
    "CoordinateSpec",
    "DimensionSpec",
    "ResolvedSchema",
    "SchemaResolver",
    # Index strategies
    "IndexStrategy",
    "RegularGridStrategy",
    "NonBinnedStrategy",
    "DuplicateHandlingStrategy",
    "ChannelWrappingStrategy",
    "ShotWrappingStrategy",
    "CompositeStrategy",
    "IndexStrategyFactory",
    # Header analysis
    "HeaderRequirements",
    "HeaderAnalyzer",
    "StreamerShotGeometryType",
    "ShotGunGeometryType",
    # Dataset building
    "DatasetFactory",
    # Coordinate utilities
    "get_coordinates",
    "get_spatial_coordinate_unit",
    "populate_coordinates",
    "update_template_units",
    # Validation
    "grid_density_qc",
    "validate_spec_in_template",
    # Metadata
    "add_grid_override_to_metadata",
    "add_segy_file_headers",
    # Pipeline
    "run_segy_ingestion",
]
