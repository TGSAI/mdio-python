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
        "Passing `grid_overrides` as a dict is deprecated and will be removed in a "
        "future release; pass a `mdio.GridOverrides` instance instead."
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
            instance; ``dict`` is still accepted but emits a :class:`DeprecationWarning`.
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
