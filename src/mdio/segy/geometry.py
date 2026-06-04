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
        """Require the parameters ``non_binned`` depends on.

        ``chunksize`` and ``non_binned_dims`` are only meaningful when collapsing dims into a
        ``trace`` axis. Enforcing the dependency on the model means every construction path
        (typed instance or a coerced legacy dict) fails fast with the same error, so the
        ingestion pipeline does not need to re-check it.

        Raises:
            GridOverrideMissingParameterError: When ``non_binned`` is set without both
                ``chunksize`` and ``non_binned_dims``.
        """
        if not self.non_binned:
            return self
        missing: set[str] = set()
        if self.chunksize is None:
            missing.add("chunksize")
        if not self.non_binned_dims:
            missing.add("non_binned_dims")
        if missing:
            command = "NonBinned"
            raise GridOverrideMissingParameterError(command, missing)
        return self

    def __bool__(self) -> bool:
        """Return True if any override flag is enabled."""
        return (
            self.auto_channel_wrap
            or self.auto_shot_wrap
            or self.calculate_shot_index
            or self.non_binned
            or self.has_duplicates
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        """Dump to the legacy ``CamelCase`` dict shape stored in dataset metadata."""
        return self.model_dump(by_alias=True, exclude_defaults=True)


def _resolve_synthesize_dims(template: AbstractDatasetTemplate | None) -> tuple[str, ...]:
    """Return dimension fields to synthesize when missing for a given template.

    Only the OBN receiver gathers template currently synthesizes ``component``; every
    other template returns ``()`` so the strategy registry skips synthesis entirely.
    """
    if template is None:
        return ()
    # Lazy import: builder templates pull in builder schemas that indirectly import this
    # module's ``GridOverrides``, so a top-level import would cycle.
    from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate  # noqa: PLC0415

    if isinstance(template, Seismic3DObnReceiverGathersTemplate):
        return ("component",)
    return ()


def validate_overrides_for_template(
    config: GridOverrides | None,
    template: AbstractDatasetTemplate | None,
) -> None:
    """Reject grid override / template pairings that v1.1 forbade.

    ``auto_shot_wrap`` is streamer-only and ``calculate_shot_index`` is OBN-only; using
    either with the wrong template silently produced wrong shot indices in v1.1 unless
    the per-command validator caught it. This is the one guard the :class:`GridOverrides`
    model cannot enforce on its own (it depends on the chosen template), so the ingestion
    pipeline calls it before any header parsing.

    Args:
        config: Typed grid overrides, or ``None`` when no overrides were requested.
        template: Template chosen by the caller, or ``None`` if omitted.

    Raises:
        TypeError: When ``auto_shot_wrap`` is set without a streamer template, or
            ``calculate_shot_index`` is set without an OBN receiver-gathers template.
    """
    if not config:
        return

    if config.auto_shot_wrap:
        # Lazy import: see ``_resolve_synthesize_dims`` for the cycle rationale.
        from mdio.builder.templates.seismic_3d_streamer_field import (  # noqa: PLC0415
            Seismic3DStreamerFieldRecordsTemplate,
        )

        if not isinstance(template, Seismic3DStreamerFieldRecordsTemplate):
            actual = type(template).__name__ if template is not None else "None"
            msg = (
                f"auto_shot_wrap only supports Seismic3DStreamerFieldRecordsTemplate, "
                f"got {actual}. For OBN templates, use calculate_shot_index."
            )
            raise TypeError(msg)

    if config.calculate_shot_index:
        from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate  # noqa: PLC0415

        if not isinstance(template, Seismic3DObnReceiverGathersTemplate):
            actual = type(template).__name__ if template is not None else "None"
            msg = f"calculate_shot_index only supports Seismic3DObnReceiverGathersTemplate, got {actual}."
            raise TypeError(msg)
