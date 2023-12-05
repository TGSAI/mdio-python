"""Variable model for MDIO V0."""


from pydantic import Field
from pydantic import create_model

from mdio.schemas import NamedArray
from mdio.schemas import ScalarType
from mdio.schemas import StructuredType
from mdio.schemas.base.encoding import ChunkGridMetadata
from mdio.schemas.base.metadata import MetadataContainer
from mdio.schemas.base.metadata import UserAttributes
from mdio.schemas.core import model_fields


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
