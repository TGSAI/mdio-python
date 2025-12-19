"""Common optimization utilities."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any

from mdio.builder.schemas.compressors import ZFP
from mdio.builder.schemas.compressors import ZFPMode

if TYPE_CHECKING:
    from collections.abc import Generator

    from xarray import DataArray

    from mdio.builder.schemas.v1.stats import SummaryStatistics


try:
    import distributed
except ImportError:
    distributed = None


logger = logging.getLogger(__name__)


class ZfpQuality(float, Enum):
    """Config options for ZFP compression."""

    VERY_LOW = 6
    LOW = 3
    MEDIUM = 1
    HIGH = 0.1
    VERY_HIGH = 0.01
    ULTRA = 0.001


def get_default_zfp(
    stats: SummaryStatistics,
    quality: ZfpQuality = ZfpQuality.LOW,
) -> ZFP:
    """Compute ZFP encoding based on data statistics and quality level."""
    if stats.std is None or stats.std <= 0:
        msg = "Standard deviation must be positive for tolerance calculation."
        raise ValueError(msg)

    tolerance = quality.value * stats.std
    logger.info("Computed ZFP tolerance: %s (quality: %s, std: %s)", tolerance, quality.name, stats.std)
    return ZFP(mode=ZFPMode.FIXED_ACCURACY, tolerance=tolerance)


def apply_compressor_encoding(
    data_array: DataArray, chunks: tuple[int, ...], zfp_encoding: dict[str, Any]
) -> DataArray:
    """Apply ZFP encoding and custom chunks to a DataArray copy."""
    # Drop coordinates to avoid re-writing them and avoid rechunking issues in views
    data_array = data_array.copy().reset_coords(drop=True)
    data_array.encoding.update(zfp_encoding)
    data_array.encoding["chunks"] = chunks
    return data_array


@contextmanager
def get_or_create_client(n_workers: int, threads_per_worker: int) -> Generator[distributed.Client, None, None]:
    """Get or create a Dask Distributed Client."""
    if distributed is None:
        msg = "The 'distributed' package is required for processing. Install: 'pip install multidimio[distributed]'."
        raise ImportError(msg)

    created = False
    try:
        client = distributed.Client.current()
        logger.info("Using existing Dask client: %s", client)
    except ValueError:
        logger.info("No active Dask client found. Creating a new one.")
        client = distributed.Client(n_workers=n_workers, threads_per_worker=threads_per_worker)
        created = True
    try:
        yield client
    finally:
        if created:
            client.close()
