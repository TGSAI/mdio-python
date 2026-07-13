"""Conversion from SEG-Y to MDIO v1 format."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mdio.segy.geometry import GridOverrides

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from segy.config import SegyHeaderOverrides
    from segy.schema import SegySpec
    from upath import UPath

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.core.dimension import Dimension


def _coerce_grid_overrides(
    grid_overrides: GridOverrides | dict[str, Any] | None,
) -> GridOverrides | None:
    """Normalize public ``grid_overrides`` input into a :class:`GridOverrides` model.

    The internal ingestion pipeline only accepts the typed model. A legacy ``dict`` is
    converted and a deprecation message is logged.
    """
    if grid_overrides is None:
        return None

    if isinstance(grid_overrides, GridOverrides):
        return grid_overrides

    logger.warning(
        "Passing `grid_overrides` as a dict is deprecated as of 1.2 and is planned for removal "
        "in a future release; pass a `mdio.GridOverrides` instance instead."
    )
    return GridOverrides.model_validate(grid_overrides)


def segy_to_mdio(  # noqa: PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: GridOverrides | dict[str, Any] | None = None,
    segy_header_overrides: SegyHeaderOverrides | None = None,
) -> None:
    """A function that converts a SEG-Y file to an MDIO v1 file.

    Ingest a SEG-Y file according to the segy_spec. This could be a spec from registry or custom.

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        mdio_template: The MDIO template to use for the conversion.
        input_path: The universal path of the input SEG-Y file.
        output_path: The universal path for the output MDIO v1 file.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.
        grid_overrides: Option to add grid overrides. Prefer a :class:`mdio.GridOverrides`
            instance; ``dict`` is still accepted (deprecated as of 1.2, planned for removal in
            a future release) but logs a deprecation warning.
        segy_header_overrides: Option to override specific SEG-Y headers during ingestion.
    """
    typed_grid_overrides = _coerce_grid_overrides(grid_overrides)

    from mdio.ingestion.segy.pipeline import segy_to_mdio as _ingest_segy_to_mdio  # noqa: PLC0415

    return _ingest_segy_to_mdio(
        segy_spec=segy_spec,
        mdio_template=mdio_template,
        input_path=input_path,
        output_path=output_path,
        overwrite=overwrite,
        grid_overrides=typed_grid_overrides,
        segy_header_overrides=segy_header_overrides,
    )


def allocate_mdio_grid(  # noqa: PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    global_dimensions: list[Dimension],
    output_path: UPath | Path | str,
    reference_segy_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: GridOverrides | dict[str, Any] | None = None,
) -> UPath:
    """Allocate an empty global MDIO store for multi-shard consolidation.

    ADDITIVE: creates the dataset skeleton sized to ``global_dimensions`` (the union of
    all shards' coordinates, including the trailing sample dimension) so shards can be
    written into sub-regions with :func:`append_segy_shard`. Does not alter single-file
    ``segy_to_mdio`` behavior.

    Args:
        segy_spec: SEG-Y spec shared by all shards.
        mdio_template: MDIO template shared by all shards.
        global_dimensions: Ordered global grid dimensions (incl. sample dimension).
        output_path: Output MDIO store path.
        reference_segy_path: Any one shard, used only to derive store units/metadata.
        overwrite: Whether to overwrite an existing store.
        grid_overrides: Optional grid overrides (dict accepted but deprecated).

    Returns:
        The normalized output path of the allocated store.
    """
    typed_grid_overrides = _coerce_grid_overrides(grid_overrides)

    from mdio.ingestion.segy.consolidate import allocate_mdio_grid as _allocate  # noqa: PLC0415

    return _allocate(
        segy_spec=segy_spec,
        mdio_template=mdio_template,
        global_dimensions=global_dimensions,
        output_path=output_path,
        reference_segy_path=reference_segy_path,
        overwrite=overwrite,
        grid_overrides=typed_grid_overrides,
    )


def append_segy_shard(  # noqa: PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    global_dimensions: list[Dimension],
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    grid_overrides: GridOverrides | dict[str, Any] | None = None,
    segy_header_overrides: SegyHeaderOverrides | None = None,
    merge_chunks: set[tuple[int, ...]] | None = None,
) -> dict[str, slice]:
    """Ingest one SEG-Y shard into its region of a pre-allocated global MDIO store.

    ADDITIVE: the store must already exist (see :func:`allocate_mdio_grid`). The shard's
    traces are placed at their global grid positions via an in-place (``mode="r+"``)
    write; other regions are untouched. Call once per shard.

    Args:
        segy_spec: SEG-Y spec (same as allocation).
        mdio_template: MDIO template (same as allocation).
        global_dimensions: Global grid dimensions used at allocation (incl. sample dim).
        input_path: The shard SEG-Y path.
        output_path: The pre-allocated global MDIO store path.
        grid_overrides: Optional grid overrides (dict accepted but deprecated).
        segy_header_overrides: Optional SEG-Y header overrides for this shard.
        merge_chunks: Optional set of shared chunk-grid indices to write read-modify-write
            (e.g. ``plan.merge_chunks_for(shard_id)`` from :func:`mdio.plan_consolidation`).

    Returns:
        The spatial region (dim name -> slice) the shard was written into.
    """
    typed_grid_overrides = _coerce_grid_overrides(grid_overrides)

    from mdio.ingestion.segy.consolidate import append_segy_shard as _append  # noqa: PLC0415

    return _append(
        segy_spec=segy_spec,
        mdio_template=mdio_template,
        global_dimensions=global_dimensions,
        input_path=input_path,
        output_path=output_path,
        grid_overrides=typed_grid_overrides,
        segy_header_overrides=segy_header_overrides,
        merge_chunks=merge_chunks,
    )
