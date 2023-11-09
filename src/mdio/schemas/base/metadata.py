"""Metadata schemas and conventions."""


import re
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


def follows_metadata_key_convention(key: str) -> bool:
    """Check if the metadata convention key meets the required format.

    We expect metadata conventions to be in the following format:

    ```JSON
    {"units-v1": {"length": "m"}}
    // or
    {"units-v1": {"length": "m"}}
    // or
    {"units-v1": {"length": "m"}}
    //or
    {"units-v1": {"length": "m"}}
    ```

    In most programming languages, hyphens are not permitted within variable
    names. Instead, underscores are used. When such a field needs to be
    externally represented with a hyphen, an alias must be for the variable,
    substituting the underscore with a hyphen.

    Args:
        key: The key to be checked against the metadata key convention.

    Returns:
        True if the key follows the metadata key convention, False otherwise.
    """
    pattern = r"^[a-z]+_v\d+$"
    return bool(re.match(pattern, key))


class VersionedMetadataConvention(BaseModel):
    """Data model for versioned metadata convention."""

    model_config = ConfigDict(
        extra="ignore",
        str_to_lower=True,
        populate_by_name=True,
        frozen=True,
        alias_generator=lambda x: x.replace("_", "-"),
    )

    @model_validator(mode="after")
    def check_key_and_alias(self):
        """Checks if key and alias matches expected patterns."""
        if len(self.model_fields) != 1:
            raise ValueError(f"{self.__class__.__name__} can only have one field.")

        name = self.__class__.__name__
        key = next(iter(self.model_fields))

        if not follows_metadata_key_convention(key):
            raise ValueError(
                f"The provided key for {name} does not follow the required "
                "naming convention. Please ensure the key is formatted as "
                "'name_v1'. For instance, use 'unit_v1'."
            )

        return self

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Override default model dump to use alias."""
        return super().model_dump(by_alias=True, *args, **kwargs)

    def model_dump_json(self, *args, **kwargs) -> str:
        """Override default model dump json to use alias."""
        return super().model_dump_json(by_alias=True, *args, **kwargs)


class UserAttributes(BaseModel):
    """User defined attributes as key/value pairs."""

    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="User defined attributes as key/value pairs.",
    )
