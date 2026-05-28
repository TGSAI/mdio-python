"""SEG-Y grid override configuration model and legacy executor shim.

The Pydantic :class:`GridOverrides` model is the supported public API for configuring
grid overrides. The :class:`GridOverrider` class is retained as a thin shim that
delegates to :class:`mdio.ingestion.index_strategies.IndexStrategyRegistry`; it preserves
the v1.1 ``run(...)`` contract for callers that still pass a legacy ``dict``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from mdio.ingestion.index_strategies import IndexStrategyRegistry
from mdio.segy.exceptions import GridOverrideMissingParameterError
from mdio.segy.exceptions import GridOverrideUnknownError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from segy.arrays import HeaderArray

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
        """Dump to the legacy ``CamelCase`` dict shape consumed by :class:`GridOverrider`."""
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


def _validate_template_for_overrides(
    config: GridOverrides,
    template: AbstractDatasetTemplate | None,
) -> None:
    """Reject grid override / template pairings that v1.1 forbade.

    ``auto_shot_wrap`` is streamer-only and ``calculate_shot_index`` is OBN-only; using
    either with the wrong template silently produced wrong shot indices in v1.1 unless
    the per-command validator caught it. This function restores that guard.

    Args:
        config: Typed grid overrides extracted from the user's legacy dict.
        template: Template chosen by the caller, or ``None`` if omitted.

    Raises:
        TypeError: When ``auto_shot_wrap`` is set without a streamer template, or
            ``calculate_shot_index`` is set without an OBN receiver-gathers template.
    """
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


class GridOverrider:
    """Legacy facade that adapts the dict-based v1.1 API onto :class:`IndexStrategyRegistry`.

    Existing callers (notably :func:`mdio.segy.utilities.get_grid_plan`) still build a
    legacy ``dict`` of grid overrides and call :meth:`run`. This class translates the dict
    into a typed :class:`GridOverrides`, dispatches to the appropriate
    :class:`IndexStrategy`, and returns the ``(headers, names, chunksize)`` tuple shape
    those callers depend on. It will be removed once all callers move to the typed API.
    """

    def __init__(self) -> None:
        self._registry = IndexStrategyRegistry()

    def run(
        self,
        index_headers: HeaderArray,
        index_names: Sequence[str],
        grid_overrides: dict[str, Any] | None,
        chunksize: Sequence[int] | None = None,
        template: AbstractDatasetTemplate | None = None,
    ) -> tuple[HeaderArray, tuple[str, ...], tuple[int, ...] | None]:
        """Run the configured grid overrides and return updated headers/names/chunks.

        Args:
            index_headers: Parsed SEG-Y trace headers; structured numpy array.
            index_names: Names of the index dimensions before any override is applied.
            grid_overrides: Legacy dict of overrides (CamelCase keys).
            chunksize: Optional chunk shape that may be expanded by overrides that add a
                ``trace`` dimension.
            template: Optional dataset template; used to identify coordinate fields and
                to drive component synthesis for OBN.

        Returns:
            Tuple of ``(transformed_headers, new_index_names, new_chunksize)``. The
            chunksize tuple is ``None`` when the caller did not pass a chunksize.

        Raises:
            GridOverrideUnknownError: When ``grid_overrides`` contains an unknown key.
            GridOverrideMissingParameterError: When ``NonBinned`` is enabled without
                ``chunksize`` or ``non_binned_dims``.

        Notes:
            Header-precondition checks (``GridOverrideKeysError``) are delegated to
            :meth:`IndexStrategy.validate_headers`; template-compatibility checks
            (``TypeError``) are delegated to :func:`_validate_template_for_overrides`.
        """
        grid_overrides = grid_overrides or {}

        field_names = set(GridOverrides.model_fields.keys())
        aliases = {field.alias for field in GridOverrides.model_fields.values() if field.alias}
        valid_keys = field_names | aliases
        for key in grid_overrides:
            if key not in valid_keys:
                raise GridOverrideUnknownError(key)

        config = GridOverrides.model_validate(grid_overrides)

        if config.non_binned:
            missing: set[str] = set()
            if config.chunksize is None:
                missing.add("chunksize")
            if not config.non_binned_dims:
                missing.add("non_binned_dims")
            if missing:
                command = "NonBinned"
                raise GridOverrideMissingParameterError(command, missing)

        _validate_template_for_overrides(config, template)

        synthesize_dims = _resolve_synthesize_dims(template)
        strategy = self._registry.create_strategy(
            grid_overrides=config,
            synthesize_dims=synthesize_dims,
            template=template,
        )
        logger.debug("Selected grid override strategy: %s", strategy.name)

        strategy.validate_headers(index_headers)
        new_headers = strategy.transform_headers(index_headers)

        new_names = list(index_names)
        new_chunks = list(chunksize) if chunksize is not None else None

        # Both NonBinned and HasDuplicates add a 'trace' dim at index -1; HasDuplicates
        # always uses chunksize 1, NonBinned uses the user-supplied value.
        if config.non_binned or config.has_duplicates:
            new_names.append("trace")
            if new_chunks is not None:
                inserted_chunk = config.chunksize if config.non_binned else 1
                new_chunks.insert(-1, inserted_chunk)

        return (
            new_headers,
            tuple(new_names),
            tuple(new_chunks) if new_chunks is not None else None,
        )
