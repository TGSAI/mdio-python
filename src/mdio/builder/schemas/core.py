"""This module implements the core components of the MDIO schemas."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel


class CamelCaseStrictModel(BaseModel):
    """A model with forbidden extras and camel case aliases."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        serialize_by_alias=True,
        validate_assignment=True,
        extra="forbid",
    )
