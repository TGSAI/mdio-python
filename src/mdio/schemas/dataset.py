"""This module provides a base class for MDIO datasets."""


from pydantic import ConfigDict
from pydantic.json_schema import GenerateJsonSchema

from mdio.schemas.base.core import StrictCamelBaseModel


JSON_SCHEMA_DIALECT = GenerateJsonSchema.schema_dialect


class BaseDataset(StrictCamelBaseModel):
    """A base class for MDIO datasets.

    We add schema dialect to extend the config of `StrictCamelBaseModel`.
    We use the default Pydantic schema generator `GenerateJsonSchema` to
    define the JSON schema dialect accurately.
    """

    model_config = ConfigDict(json_schema_extra={"$schema": JSON_SCHEMA_DIALECT})
