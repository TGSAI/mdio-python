"""Dataset model for MDIO V0."""


from pydantic import Field

from mdio.schemas.base.core import StrictCamelBaseModel
from mdio.schemas.base.metadata import UserAttributes
from mdio.schemas.v0.variable import Variable


class Dataset(StrictCamelBaseModel):
    """Represents an MDIO dataset.

    A dataset consists of variables and metadata.
    """

    name: str = Field(..., description="Name of the dataset.")
    variables: list[Variable] = Field(..., description="Variables in MDIO dataset")
    attributes: UserAttributes | None = Field(
        default=None, description="Dataset metadata."
    )
