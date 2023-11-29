"""Metadata schemas and conventions."""


import re
from typing import Any

from pydantic import Field

from mdio.schemas.base.core import StrictCamelBaseModel


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


class VersionedMetadataConvention(StrictCamelBaseModel):
    """Data model for versioned metadata convention."""

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Override default model dump to use alias."""
        return super().model_dump(*args, **kwargs, by_alias=True)

    def model_dump_json(self, *args, **kwargs) -> str:
        """Override default model dump json to use alias."""
        return super().model_dump_json(*args, **kwargs, by_alias=True)


class UserAttributes(StrictCamelBaseModel):
    """User defined attributes as key/value pairs."""

    attributes: dict[str, Any] | None = Field(
        default=None,
        description="User defined attributes as key/value pairs.",
    )


class MetadataContainer(StrictCamelBaseModel):
    """A container model for metadata."""
