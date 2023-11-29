"""Dataset model for MDIO V1."""


from pydantic import Field

from mdio.schemas.base.core import StrictCamelBaseModel
from mdio.schemas.base.dimension import DimensionCollection
from mdio.schemas.base.metadata import UserAttributes
from mdio.schemas.v1.variable import Variable


class Dataset(StrictCamelBaseModel):
    """Represents an MDIO dataset.

    A dataset consists of variables and metadata.
    """

    name: str = Field(..., description="Name of the dataset.")
    variables: list[Variable] = Field(..., description="Variables in MDIO dataset")
    dimensions: DimensionCollection | None = Field(
        default=None,
        description="List of Dimension collection or reference to dimension names.",
    )
    metadata: UserAttributes | None = Field(
        default=None, description="Dataset metadata."
    )
