"""Module that contains type aliases for templates."""

from typing import Literal
from typing import TypeAlias

SeismicDataDomain: TypeAlias = Literal["depth", "time"]

CdpGatherDomain: TypeAlias = Literal["offset", "angle"]
