"""Statistics schema for MDIO v1 arrays.

This module provides two Histogram classes (CenteredBinHistogram and
EdgeDefinedHistogram),a summary statistics class, and a summary statistics
metadata class.

SummaryStatistics: a class that represents the minimum summary statistics
of an array consisting of count, sum, sum of squares, min, max, and a histogram.

SummaryStatisticsMetadata: represents metadata for statistics, with a field
for v1 of the stats.

CenteredBinHistogram takes the center points of each bin in a histogram,
while EdgeDefinedHistogram takes the left edges and widths of each bin.
Both classes extend from the base class BaseHistogram, which represents
a histogram with count of each bin.
"""

from typing import TypeAlias

from pydantic import Field

from mdio.schemas.base.metadata import VersionedMetadataConvention
from mdio.schemas.core import StrictCamelBaseModel


class BaseHistogram(StrictCamelBaseModel):
    """Represents a histogram with bin counts."""

    counts: list[int] = Field(..., description="Count of each each bin.")


class CenteredBinHistogram(BaseHistogram):
    """Class representing a center bin histogram."""

    bin_centers: list[float | int] = Field(..., description="List of bin centers.")


class EdgeDefinedHistogram(BaseHistogram):
    """A class representing an edge-defined histogram."""

    bin_edges: list[float | int] = Field(
        ..., description="The left edges of the histogram bins."
    )
    bin_widths: list[float | int] = Field(
        ..., description="The widths of the histogram bins."
    )


Histogram: TypeAlias = CenteredBinHistogram | EdgeDefinedHistogram


class SummaryStatistics(StrictCamelBaseModel):
    """Data model for some statistics in MDIO v1 arrays."""

    count: int = Field(..., description="The number of data points.")
    sum: float = Field(..., description="The total of all data values.")
    sum_squares: float = Field(..., description="The total of all data values squared.")
    min: float = Field(..., description="The smallest value in the variable.")
    max: float = Field(..., description="The largest value in the variable.")
    histogram: Histogram = Field(..., description="Binned frequency distribution.")


class StatisticsMetadata(VersionedMetadataConvention):
    """Data Model representing metadata for statistics."""

    stats_v1: SummaryStatistics | list[SummaryStatistics] | None = Field(
        default=None,
        description="Minimal summary statistics.",
    )
