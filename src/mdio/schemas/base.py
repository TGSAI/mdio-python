"""Base models to subclass from."""


from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel
from pydantic.json_schema import GenerateJsonSchema


JSON_SCHEMA_DIALECT = GenerateJsonSchema.schema_dialect


class StrictCamelBaseModel(BaseModel):
    """A BaseModel subclass with Pascal Cased aliases."""

    model_config = ConfigDict(extra="forbid", alias_generator=to_camel)


class BaseDataset(StrictCamelBaseModel):
    """A base class for MDIO datasets.

    We add schema dialect to extend the config of `StrictCamelBaseModel`.
    We use the default Pydantic schema generator `GenerateJsonSchema` to
    define the JSON schema dialect accurately.
    """

    model_config = ConfigDict(json_schema_extra={"$schema": JSON_SCHEMA_DIALECT})
