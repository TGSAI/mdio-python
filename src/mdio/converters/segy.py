"""Conversion from SEG-Y to MDIO v1 format."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdio.segy.geometry import GridOverrides

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from segy.config import SegyHeaderOverrides
    from segy.schema import SegySpec
    from upath import UPath

    from mdio.builder.templates.base import AbstractDatasetTemplate


def segy_to_mdio(  # noqa PLR0913
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
        grid_overrides: Grid override configuration. Can be a GridOverrides instance for type
            safety, or a dict for backward compatibility. See GridOverrides class for available
            options.
        segy_header_overrides: Option to override specific SEG-Y headers during ingestion.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
    """
    # Convert dict to GridOverrides if needed for type safety
    if isinstance(grid_overrides, dict):
        grid_overrides = GridOverrides.model_validate(grid_overrides) if grid_overrides else None

    # Use ingestion pipeline
    from mdio.ingestion.pipeline import run_segy_ingestion

    return run_segy_ingestion(
        segy_spec=segy_spec,
        mdio_template=mdio_template,
        input_path=input_path,
        output_path=output_path,
        overwrite=overwrite,
        grid_overrides=grid_overrides,
        segy_header_overrides=segy_header_overrides,
    )
