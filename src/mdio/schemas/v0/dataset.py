"""Dataset model for MDIO V0."""


from pydantic import Field

from mdio.schemas.base import BaseDataset
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v0.variable import Variable


class Dataset(BaseDataset):
    """Represents an MDIO dataset.

    A dataset consists of variables and metadata.
    """

    name: str = Field(..., description="Name of the dataset.")
    variables: list[Variable] = Field(..., description="Variables in MDIO dataset")
    attributes: UserAttributes | None = Field(
        default=None, description="Dataset metadata."
    )
