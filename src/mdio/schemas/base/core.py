"""This module implements the core components of the MDIO schemas."""


from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel


class StrictCamelBaseModel(BaseModel):
    """A BaseModel subclass with Pascal Cased aliases."""

    model_config = ConfigDict(extra="forbid", alias_generator=to_camel)
