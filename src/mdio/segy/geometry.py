"""SEG-Y grid override configuration model and template-compatibility helpers.

The Pydantic :class:`GridOverrides` model is the supported public API for configuring
grid overrides. Header transformation and schema reshaping are owned by
:class:`mdio.ingestion.segy.index_strategies.IndexStrategyRegistry`; this module only holds
the typed config plus the template-compatibility guards used to validate override pairings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from mdio.segy.exceptions import GridOverrideIncompatibleError
from mdio.segy.exceptions import GridOverrideMissingParameterError

if TYPE_CHECKING:
    from mdio.builder.templates.base import AbstractDatasetTemplate


logger = logging.getLogger(__name__)


class GridOverrides(BaseModel):
    """Type-safe configuration for grid override operations during SEG-Y ingestion."""

    model_config = ConfigDict(extra="forbid", validate_by_name=True)

    auto_channel_wrap: bool = Field(
        default=False,
        alias="AutoChannelWrap",
        description="Streamer: auto-detect channel-wrap geometry (Type A vs B).",
    )
    auto_shot_wrap: bool = Field(
        default=False,
        alias="AutoShotWrap",
        description="Streamer: derive dense shot_index from interleaved shot_point values.",
    )
    calculate_shot_index: bool = Field(
        default=False,
        alias="CalculateShotIndex",
        description="OBN: derive dense shot_index from sparse shot_point values per shot_line.",
    )
    calculate_segment_index: bool = Field(
        default=False,
        alias="CalculateSegmentIndex",
        description="Continuous recording: derive dense segment_index by ranking epoch per receiver.",
    )
    non_binned: bool = Field(
        default=False,
        alias="NonBinned",
        description="Collapse selected dims into a single trace dimension without spatial binning.",
    )
    has_duplicates: bool = Field(
        default=False,
        alias="HasDuplicates",
        description="Add a trace dimension (chunksize 1) to disambiguate duplicate trace indices.",
    )
    chunksize: int | None = Field(
        default=None,
        gt=0,
        description="Chunk size for the trace dimension when `non_binned` is True.",
    )
    non_binned_dims: list[str] | None = Field(
        default=None,
        description="Dimension names to collapse into the trace dimension when `non_binned` is True.",
    )

    @model_validator(mode="after")
    def _check_non_binned_parameters(self) -> GridOverrides:
        """Validate parameters when non_binned is True.

        Raises:
            GridOverrideMissingParameterError: If chunksize or non_binned_dims is missing.

        Returns:
            The validated GridOverrides instance.
        """
        if not self.non_binned:
            return self

        missing = set()
        if self.chunksize is None:
            missing.add("chunksize")
        if not self.non_binned_dims:
            missing.add("non_binned_dims")

        if missing:
            command = "NonBinned"
            raise GridOverrideMissingParameterError(command, missing)
        return self

    @model_validator(mode="after")
    def _check_segment_index_exclusivity(self) -> GridOverrides:
        """Reject pairing ``calculate_segment_index`` with ``non_binned`` or ``has_duplicates``.

        Raises:
            GridOverrideIncompatibleError: If incompatible overrides are combined.

        Returns:
            The validated GridOverrides instance.
        """
        if not self.calculate_segment_index:
            return self

        this_command = "CalculateSegmentIndex"
        for flag, other_command in ((self.non_binned, "NonBinned"), (self.has_duplicates, "HasDuplicates")):
            if flag:
                raise GridOverrideIncompatibleError(this_command, other_command)
        return self

    def __bool__(self) -> bool:
        """Return True if any override flag is enabled."""
        return (
            self.auto_channel_wrap
            or self.auto_shot_wrap
            or self.calculate_shot_index
            or self.calculate_segment_index
            or self.non_binned
            or self.has_duplicates
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        """Dump to the legacy ``CamelCase`` dict shape stored in dataset metadata."""
        return self.model_dump(by_alias=True, exclude_defaults=True)


def validate_overrides_for_template(
    config: GridOverrides | None,
    template: AbstractDatasetTemplate | None,
) -> None:
    """Reject grid override / template pairings that v1.1 forbade.

    ``auto_shot_wrap`` is streamer-only, ``calculate_shot_index`` is OBN-only, and
    ``calculate_segment_index`` is continuous-recording-only. These are the guards
    :class:`GridOverrides` cannot enforce on its own (they depend on the chosen template),
    so the ingestion pipeline calls it before any header parsing.

    Args:
        config: Typed grid overrides, or ``None`` when no overrides were requested.
        template: Template chosen by the caller, or ``None`` if omitted.

    Raises:
        TypeError: When an override is paired with an unsupported template.
    """
    if not config:
        return

    if config.auto_shot_wrap:
        # Lazy import: builder templates pull in builder schemas that indirectly import this
        # module's ``GridOverrides``, so a top-level import would cycle.
        from mdio.builder.templates.seismic_3d_streamer_field import (  # noqa: PLC0415
            Seismic3DStreamerFieldRecordsTemplate,
        )

        if not isinstance(template, Seismic3DStreamerFieldRecordsTemplate):
            actual = type(template).__name__ if template is not None else "None"
            msg = (
                f"auto_shot_wrap only supports {Seismic3DStreamerFieldRecordsTemplate.__name__}, "
                f"got {actual}. For OBN templates, use calculate_shot_index."
            )
            raise TypeError(msg)

    if config.calculate_shot_index:
        from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate  # noqa: PLC0415

        if not isinstance(template, Seismic3DObnReceiverGathersTemplate):
            actual = type(template).__name__ if template is not None else "None"
            msg = f"calculate_shot_index only supports {Seismic3DObnReceiverGathersTemplate.__name__}, got {actual}."
            raise TypeError(msg)

    if config.calculate_segment_index:
        from mdio.builder.templates.seismic_3d_nodal_continuous_receiver_gathers import (  # noqa: PLC0415
            Seismic3DNodalContinuousReceiverGathersTemplate,
        )

        if not isinstance(template, Seismic3DNodalContinuousReceiverGathersTemplate):
            actual = type(template).__name__ if template is not None else "None"
            msg = (
                f"calculate_segment_index only supports "
                f"{Seismic3DNodalContinuousReceiverGathersTemplate.__name__}, got {actual}."
            )
            raise TypeError(msg)
