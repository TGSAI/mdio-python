"""Metadata handling utilities for MDIO ingestion.

This module contains functions for managing metadata during SEG-Y to MDIO conversion,
including grid overrides and SEG-Y file headers.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from mdio.core.config import MDIOSettings

if TYPE_CHECKING:
    from xarray import Dataset as xr_Dataset

    from mdio.builder.schemas.v1.dataset import Dataset
    from mdio.segy.file import SegyFileInfo
    from mdio.segy.geometry import GridOverrides

logger = logging.getLogger(__name__)


def add_grid_override_to_metadata(dataset: Dataset, grid_overrides: GridOverrides | None) -> None:
    """Add grid override to Dataset metadata if needed."""
    if dataset.metadata.attributes is None:
        dataset.metadata.attributes = {}

    if grid_overrides is not None:
        dataset.metadata.attributes["gridOverrides"] = (
            grid_overrides.model_dump(by_alias=True, exclude_defaults=True, exclude={"extra_params"})
            | grid_overrides.extra_params
        )


def add_segy_file_headers(xr_dataset: xr_Dataset, segy_file_info: SegyFileInfo) -> xr_Dataset:
    """Add SEG-Y file headers to xarray dataset."""
    settings = MDIOSettings()

    if not settings.save_segy_file_header:
        return xr_dataset

    expected_rows = 40
    expected_cols = 80

    text_header_rows = segy_file_info.text_header.splitlines()
    text_header_cols_bad = [len(row) != expected_cols for row in text_header_rows]

    if len(text_header_rows) != expected_rows:
        err = f"Invalid text header count: expected {expected_rows}, got {len(segy_file_info.text_header)}"
        raise ValueError(err)

    if any(text_header_cols_bad):
        err = f"Invalid text header columns: expected {expected_cols} per line."
        raise ValueError(err)

    xr_dataset["segy_file_header"] = ((), "")
    xr_dataset["segy_file_header"].attrs.update(
        {
            "textHeader": segy_file_info.text_header,
            "binaryHeader": segy_file_info.binary_header_dict,
        }
    )
    if settings.raw_headers:
        raw_binary_base64 = base64.b64encode(segy_file_info.raw_binary_headers).decode("ascii")
        xr_dataset["segy_file_header"].attrs.update({"rawBinaryHeader": raw_binary_base64})

    return xr_dataset
