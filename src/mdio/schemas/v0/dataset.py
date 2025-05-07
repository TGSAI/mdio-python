"""Dataset model for MDIO V0."""

from __future__ import annotations

from pydantic import AwareDatetime
from pydantic import Field

from mdio.schemas.base import BaseArray
from mdio.schemas.base import BaseDataset
from mdio.schemas.core import CamelCaseStrictModel
from mdio.schemas.core import StrictModel


class DimensionModelV0(CamelCaseStrictModel):
    """Represents dimension schema for MDIO v0."""

    name: str = Field(..., description="Name of the dimension.")
    coords: list[int] = Field(..., description="Coordinate labels (ticks).")


class DatasetMetadataModelV0(StrictModel):
    """Represents dataset attributes schema for MDIO v0."""

    api_version: str = Field(
        ...,
        description="MDIO version.",
    )

    created: AwareDatetime = Field(
        ...,
        description="Creation time with TZ info.",
    )

    dimension: list[DimensionModelV0] = Field(
        ...,
        description="Dimensions.",
    )

    mean: float | None = Field(
        default=None,
        description="Mean value of the samples.",
    )

    # Statistical information
    std: float | None = Field(default=None, description="Standard deviation of the samples.")

    rms: float | None = Field(default=None, description="Root mean squared value of the samples.")

    min: float | None = Field(
        default=None,
        description="Minimum value of the samples.",
    )

    max: float | None = Field(
        default=None,
        description="Maximum value of the samples.",
    )

    trace_count: int | None = Field(default=None, description="Number of traces in the SEG-Y file.")


class VariableModelV0(BaseArray):
    """Represents an MDIO v0 variable schema."""


class DatasetModelV0(BaseDataset):
    """Represents an MDIO v0 dataset schema."""

    seismic: list[VariableModelV0] = Field(
        ...,
        description="Variable containing seismic.",
    )

    headers: list[VariableModelV0] = Field(
        ...,
        description="Variable containing headers.",
    )

    metadata: DatasetMetadataModelV0 = Field(
        ...,
        description="Dataset metadata.",
    )
