"""Arguments for SegyFile instance creation."""

from typing import TypedDict
from segy.schema import SegySpec
from segy.config import SegySettings

class SegyFileArguments(TypedDict):
    """Arguments to open SegyFile instance creation."""

    url: str
    spec: SegySpec | None
    settings: SegySettings | None