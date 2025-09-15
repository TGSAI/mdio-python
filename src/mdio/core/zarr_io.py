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
    """Context manager to suppress Zarr V3 unstable structured array warning."""
    warn = r"The data type \((.*?)\) does not have a Zarr V3 specification\."
    warnings.filterwarnings("ignore", message=warn, category=UnstableSpecificationWarning)
    try:
        yield
    finally:
        pass
