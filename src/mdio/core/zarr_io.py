"""Utilities to open/write Zarr files."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

from zarr.errors import UnstableSpecificationWarning

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def zarr_warnings_suppress_unstable_structs_v3() -> Generator[None, None, None]:
    """Context manager to suppress Zarr V3 unstable data-type warning.

    Covers unspecified v3 types such as ``raw_bytes`` (MDIO ``raw_headers`` / ``V240``).
    Filters are scoped with ``warnings.catch_warnings`` and restored on exit.
    """
    warn = r"The data type \((.*?)\) does not have a Zarr V3 specification\."
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=warn, category=UnstableSpecificationWarning)
        yield
