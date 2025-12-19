"""Optimize MDIO seismic datasets for fast access patterns using ZFP compression and Dask.

This module provides tools to create compressed, rechunked transpose views of seismic data for efficient
access along dataset dimensions. It uses configurable ZFP compression based on data statistics and
supports parallel processing with Dask Distributed.
"""

import logging

from pydantic import BaseModel
from pydantic import Field
from xarray import Dataset as xr_Dataset

from mdio import to_mdio
from mdio.builder.schemas.compressors import ZFP
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.v1.stats import SummaryStatistics
from mdio.builder.xarray_builder import _compressor_to_encoding
from mdio.optimize.common import apply_compressor_encoding
from mdio.optimize.common import get_default_zfp
from mdio.optimize.common import get_or_create_client

logger = logging.getLogger(__name__)


class OptimizedAccessPatternConfig(BaseModel):
    """Configuration for fast access pattern optimization."""

    optimize_dimensions: dict[str, tuple[int, ...]] = Field(..., description="Optimize dims and desired chunks.")
    processing_chunks: dict[str, int] = Field(..., description="Chunk sizes for processing the original variable.")
    compressor: Blosc | ZFP | None = Field(default=None, description="Compressor to use for access patterns.")


def optimize_access_patterns(
    dataset: xr_Dataset,
    config: OptimizedAccessPatternConfig,
    n_workers: int = 1,
    threads_per_worker: int = 1,
) -> None:
    """Optimize MDIO dataset for fast access along dimensions.

    Optimize an MDIO dataset by creating compressed, rechunked views for fast access along
    configurable dimensions, then append them to the existing MDIO file.

    This uses ZFP compression with tolerance based on data standard deviation and the provided quality level.
    Requires Dask Distributed for parallel execution. It will try to grab the existing distributed.Client
    or create its own. Existing Client will be kept running after optimization.

    Args:
        dataset: MDIO Dataset containing the seismic data.
        config: Configuration object with quality, access patterns, and processing chunks.
        n_workers: Number of Dask workers. Default is 1.
        threads_per_worker: Threads per Dask worker. Default is 1.

    Raises:
        ValueError: If required attrs/stats are missing or the dataset is invalid.

    Examples:
        For Post-Stack 3D seismic data, we can optimize the inline, crossline, and depth dimensions.

        >>> from mdio import optimize_access_patterns, OptimizedAccessPatternConfig
        >>> from mdio import open_mdio
        >>>
        >>> conf = OptimizedAccessPatternConfig(
        >>>     optimize_dimensions={
        >>>         "inline": (4, 512, 512),
        >>>         "crossline": (512, 4, 512),
        >>>         "time": (512, 512, 4),
        >>>     },
        >>>     processing_chunks= {"inline": 512, "crossline": 512, "time": 512}
        >>> )
        >>>
        >>> ds = open_mdio("/path/to/seismic.mdio")
        >>> optimize_access_patterns(ds, conf, n_workers=4)
    """
    # Extract and validate key attrs
    attrs = dataset.attrs.get("attributes", {})
    var_name = attrs.get("defaultVariableName")
    if not var_name:
        msg = "Default variable name is missing from dataset attributes."
        raise ValueError(msg)

    variable = dataset[var_name]
    chunked_var = variable.chunk(**config.processing_chunks, inline_array=True)

    if config.compressor is None:
        if "statsV1" not in variable.attrs:
            msg = "Statistics are missing from data. Std. dev. is required for compression."
            raise ValueError(msg)

        logger.info("No compressor provided, using default ZFP compression with MEDIUM quality.")
        stats = SummaryStatistics.model_validate_json(variable.attrs["statsV1"])
        default_zfp = get_default_zfp(stats)
        config.compressor = default_zfp

    compressor_encoding = _compressor_to_encoding(config.compressor)

    optimized_variables = {}
    for dim_name, dim_new_chunks in config.optimize_dimensions.items():
        if dim_name not in chunked_var.dims:
            msg = f"Dimension to optimize '{dim_name}' not found in original dataset dims: {chunked_var.dims}."
            raise ValueError(msg)
        optimized_var = apply_compressor_encoding(chunked_var, dim_new_chunks, compressor_encoding)
        optimized_var.name = f"fast_{dim_name}"
        optimized_variables[optimized_var.name] = optimized_var

    optimized_dataset = xr_Dataset(optimized_variables, attrs=dataset.attrs)
    source_path = dataset.encoding["source"]

    with get_or_create_client(n_workers=n_workers, threads_per_worker=threads_per_worker) as client:
        # The context manager ensures distributed is installed so we can try to register the plugin
        # safely. The plugin is conditionally created based on the installation status of distributed
        from mdio.optimize.patch import MonkeyPatchZfpDaskPlugin  # noqa: PLC0415

        client.register_plugin(MonkeyPatchZfpDaskPlugin())
        logger.info("Starting optimization with compressor %s.", compressor_encoding)
        to_mdio(optimized_dataset, source_path, mode="a")
        logger.info("Optimization completed successfully.")
