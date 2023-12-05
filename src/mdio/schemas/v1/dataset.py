"""Dataset model for MDIO V1."""


from pydantic import Field

from mdio.schemas.base.metadata import UserAttributes
from mdio.schemas.dataset import BaseDataset
from mdio.schemas.dimension import NamedDimension
from mdio.schemas.v1.variable import Variable


class Dataset(BaseDataset):
    """Represents an MDIO dataset.

    A dataset consists of variables and metadata.
    """

    name: str = Field(..., description="Name of the dataset.")
    variables: list[Variable] = Field(..., description="Variables in MDIO dataset")
    dimensions: list[NamedDimension] | None = Field(
        default=None, description="List of Dimensions."
    )
    metadata: UserAttributes | None = Field(
        default=None, description="Dataset metadata."
    )
