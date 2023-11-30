"""Variable model for MDIO V0."""


from pydantic import Field

from mdio.schemas.base.array import NamedArray
from mdio.schemas.base.dtype import ScalarType
from mdio.schemas.base.dtype import StructuredType
from mdio.schemas.base.metadata import UserAttributes


class Variable(NamedArray):
    """An MDIO variable that has coordinates and metadata."""

    data_type: ScalarType | StructuredType = Field(
        ..., description="Type of the array."
    )

    metadata: UserAttributes | None = Field(
        default=None, description="Variable metadata."
    )
