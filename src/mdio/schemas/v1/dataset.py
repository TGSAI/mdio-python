"""Dataset model for MDIO V1."""

from pydantic import AwareDatetime
from pydantic import Field
from pydantic import create_model

from mdio.schemas.base import BaseDataset
from mdio.schemas.core import CamelCaseStrictModel
from mdio.schemas.core import model_fields
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.variable import Variable


class DatasetInfo(CamelCaseStrictModel):
    """Contains information about a dataset."""

    name: str = Field(..., description="Name or identifier for the dataset.")

    api_version: str = Field(
        ...,
        description="The version of the MDIO API that the dataset complies with.",
    )

    created_on: AwareDatetime = Field(
        ...,
        description=(
            "The timestamp indicating when the dataset was first created, "
            "including timezone information. Expressed in ISO 8601 format."
        ),
    )


DatasetMetadata = create_model(
    "DatasetMetadata",
    **model_fields(DatasetInfo),
    **model_fields(UserAttributes),
    __base__=CamelCaseStrictModel,
)
DatasetMetadata.__doc__ = "The metadata about the dataset."


class Dataset(BaseDataset):
    """Represents an MDIO v1 dataset.

    A dataset consists of variables and metadata.
    """

    variables: list[Variable] = Field(..., description="Variables in MDIO dataset")
    metadata: DatasetMetadata = Field(..., description="Dataset metadata.")
