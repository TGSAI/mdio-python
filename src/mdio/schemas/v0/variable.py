"""Variable model for MDIO V0."""


from pydantic import Field
from pydantic import create_model

from mdio.schemas.base import NamedArray
from mdio.schemas.core import model_fields
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import MetadataContainer
from mdio.schemas.metadata import UserAttributes


VariableMetadata = create_model(
    "VariableMetadata",
    **model_fields(ChunkGridMetadata),
    **model_fields(UserAttributes),
    __base__=MetadataContainer,
)


class Variable(NamedArray):
    """An MDIO variable that has coordinates and metadata."""

    data_type: ScalarType | StructuredType = Field(
        ..., description="Type of the array."
    )

    metadata: VariableMetadata | None = Field(
        default=None, description="Variable metadata."
    )
