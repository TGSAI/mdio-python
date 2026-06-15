"""Attach SEG-Y text and binary file headers to the dataset."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from mdio.core.config import SAVE_SEGY_FILE_HEADER_LENIENT
from mdio.core.config import SAVE_SEGY_FILE_HEADER_OFF
from mdio.core.config import SAVE_SEGY_FILE_HEADER_STRICT
from mdio.core.config import MDIOSettings
from mdio.segy.text_header import sanitize_text_header
from mdio.segy.text_header import validate_text_header

if TYPE_CHECKING:
    from xarray import Dataset as xr_Dataset

    from mdio.segy.file import SegyFileInfo


logger = logging.getLogger(__name__)


def add_segy_file_headers(xr_dataset: xr_Dataset, segy_file_info: SegyFileInfo) -> xr_Dataset:
    """Attach the SEG-Y text and binary file headers as attrs on a scalar variable."""
    settings = MDIOSettings()
    mode = settings.save_segy_file_header

    if mode == SAVE_SEGY_FILE_HEADER_OFF:
        return xr_dataset

    text_header = segy_file_info.text_header

    if mode == SAVE_SEGY_FILE_HEADER_LENIENT:
        try:
            validate_text_header(text_header)
        except ValueError as exc:
            logger.warning("Correcting malformed SEG-Y text header on import: %s", exc)
        text_header = sanitize_text_header(text_header)
    elif mode == SAVE_SEGY_FILE_HEADER_STRICT:
        validate_text_header(text_header)

    xr_dataset["segy_file_header"] = ((), "")
    xr_dataset["segy_file_header"].attrs.update(
        {
            "textHeader": text_header,
            "binaryHeader": segy_file_info.binary_header_dict,
        }
    )
    if settings.raw_headers:
        raw_binary_base64 = base64.b64encode(segy_file_info.raw_binary_headers).decode("ascii")
        xr_dataset["segy_file_header"].attrs.update({"rawBinaryHeader": raw_binary_base64})

    return xr_dataset
