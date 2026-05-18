"""Attach SEG-Y text and binary file headers to the dataset."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from mdio.core.config import MDIOSettings

if TYPE_CHECKING:
    from xarray import Dataset as xr_Dataset

    from mdio.segy.file import SegyFileInfo


def _add_segy_file_headers(xr_dataset: xr_Dataset, segy_file_info: SegyFileInfo) -> xr_Dataset:
    """Attach the SEG-Y text and binary file headers as attrs on a scalar variable."""
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
