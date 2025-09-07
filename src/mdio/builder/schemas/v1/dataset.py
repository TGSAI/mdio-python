"""Dataset model for MDIO V1."""

from typing import Any

from pydantic import AwareDatetime
from pydantic import Field

from mdio.builder.schemas.base import BaseDataset
from mdio.builder.schemas.core import CamelCaseStrictModel
from mdio.builder.schemas.v1.variable import Variable


class DatasetMetadata(CamelCaseStrictModel):
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

    attributes: dict[str, Any] | None = Field(default=None, description="User defined attributes as key/value pairs.")


class Dataset(BaseDataset):
    """Represents an MDIO v1 dataset.

    A dataset consists of variables and metadata.
    """

    variables: list[Variable] = Field(..., description="Variables in MDIO dataset")
    metadata: DatasetMetadata = Field(..., description="Dataset metadata.")
