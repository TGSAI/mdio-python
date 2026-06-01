"""Type aliases and declarative specs for templates."""

from typing import Literal
from typing import TypeAlias

from pydantic import BaseModel

from mdio.builder.schemas.dtype import ScalarType

SeismicDataDomain: TypeAlias = Literal["depth", "time"]

CdpGatherDomain: TypeAlias = Literal["offset", "angle"]


class CoordinateSpec(BaseModel):
    """Specification for a non-dimension coordinate declared by a template.

    Templates declare their non-dimension coordinates via
    :meth:`AbstractDatasetTemplate.declare_coordinate_specs`. The ingestion
    ``SchemaResolver`` consumes these specs to build the final resolved schema.

    Attributes:
        name: Coordinate name (e.g. ``"cdp_x"``, ``"gun"``, ``"source_coord_x"``).
        dimensions: Names of the dimensions this coordinate is indexed by.
        dtype: Data type for the coordinate.
    """

    name: str
    dimensions: tuple[str, ...]
    dtype: ScalarType
