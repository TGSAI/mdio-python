"""This module defines two data models - Statistics and StatisticsMetadata.

- The Statistics model defines statistical indicators related to set of samples.
  These indicators include the mean, standard deviation, root mean squared,
  minimum and maximum values of the samples.

- The StatisticsMetadata model represents metadata related to these statistics.
  This metadata includes a version-specific stats field (`stats_v0`).
"""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from mdio.schemas.base.metadata import VersionedMetadataConvention


class Statistics(BaseModel):
    """Data model for some statistics in MDIO v0."""

    model_config = ConfigDict(extra="forbid")

    mean: float = Field(..., description="Mean value of the samples.")
    std: float = Field(..., description="Standard deviation of the samples.")
    rms: float = Field(..., description="Root mean squared value of the samples.")
    min: float = Field(..., description="Minimum value of the samples.")
    max: float = Field(..., description="Maximum value of the samples.")


class StatisticsMetadata(VersionedMetadataConvention):
    """Data Model representing metadata for statistics."""

    model_config = ConfigDict(extra="forbid")

    stats_v0 = Field(..., alias="stats-v0")
