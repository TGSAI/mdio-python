"""Unit tests for optimize_access_pattern module."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from distributed import Client
from segy import SegyFactory
from segy.standards import get_segy_standard
from zarr.codecs import ZFPY as zarr_ZFPY  # noqa: N811
from zarr.codecs import BloscCodec as zarr_BloscCodec

from mdio import open_mdio
from mdio import segy_to_mdio
from mdio.builder.schemas.compressors import Blosc as mdio_Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.template_registry import get_template
from mdio.optimize.access_pattern import OptimizedAccessPatternConfig
from mdio.optimize.access_pattern import optimize_access_patterns

if TYPE_CHECKING:
    from pathlib import Path


INLINES = np.arange(1, 9)
CROSSLINES = np.arange(1, 17)
NUM_SAMPLES = 64

SPEC = get_segy_standard(1)


@pytest.fixture(scope="module")
def test_segy_path(fake_segy_tmp: Path) -> Path:
    """Create a small synthetic 3D SEG-Y file."""
    segy_path = fake_segy_tmp / "optimize_ap_test_3d.sgy"

    num_traces = len(INLINES) * len(CROSSLINES)

    factory = SegyFactory(spec=SPEC, sample_interval=4000, samples_per_trace=NUM_SAMPLES)
    headers = factory.create_trace_header_template(num_traces)
    samples = factory.create_trace_sample_template(num_traces)

    headers["inline"] = INLINES.repeat(len(CROSSLINES))
    headers["crossline"] = np.tile(CROSSLINES, len(INLINES))
    headers["coordinate_scalar"] = 1

    samples[:] = np.arange(num_traces)[..., None]

    with segy_path.open(mode="wb") as fp:
        fp.write(factory.create_textual_header())
        fp.write(factory.create_binary_header())
        fp.write(factory.create_traces(headers, samples))

    return segy_path


@pytest.fixture(scope="module")
def mdio_dataset_path(test_segy_path: Path, zarr_tmp: Path) -> Path:
    """Convert synthetic SEG-Y to MDIO."""
    test_mdio_path = zarr_tmp / "optimize_ap_test_3d.mdio"

    env = {
        "MDIO__IMPORT__CPU_COUNT": "true",
        "MDIO__IMPORT__CLOUD_NATIVE": "true",
    }
    patch.dict(os.environ, env)
    segy_to_mdio(
        segy_spec=SPEC,
        mdio_template=get_template("PostStack3DTime"),
        input_path=test_segy_path,
        output_path=test_mdio_path,
        overwrite=True,
    )
    return test_mdio_path


class TestOptimizeAccessPattern:
    """Tests for optimize_access_pattern module."""

    def test_optimize_access_patterns(self, mdio_dataset_path: str) -> None:
        """Test optimization of access patterns."""
        conf = OptimizedAccessPatternConfig(
            optimize_dimensions={"time": (128, 128, 4), "inline": (2, 64, 64)},
            processing_chunks={"inline": 128, "crossline": 128, "time": 128},
        )
        ds = open_mdio(mdio_dataset_path)
        optimize_access_patterns(ds, conf)

        ds = open_mdio(mdio_dataset_path)

        assert "fast_time" in ds.variables
        assert ds["fast_time"].encoding["chunks"] == (128, 128, 4)
        assert isinstance(ds["fast_time"].encoding["serializer"], zarr_ZFPY)

        assert "inline" in ds.variables
        assert ds["fast_inline"].encoding["chunks"] == (2, 64, 64)
        assert isinstance(ds["fast_inline"].encoding["serializer"], zarr_ZFPY)

    def test_optimize_access_patterns_custom_compressor(self, mdio_dataset_path: str) -> None:
        """Test optimization of access patterns with custom compressor."""
        conf = OptimizedAccessPatternConfig(
            optimize_dimensions={"crossline": (32, 8, 32)},
            processing_chunks={"inline": 512, "crossline": 512, "time": 512},
            compressor=mdio_Blosc(cname=BloscCname.blosclz, clevel=1),
        )
        ds = open_mdio(mdio_dataset_path)
        optimize_access_patterns(ds, conf)

        ds = open_mdio(mdio_dataset_path)

        actual_compressor = ds["fast_crossline"].encoding["compressors"][0]
        assert "fast_crossline" in ds.variables
        assert ds["fast_crossline"].encoding["chunks"] == (32, 8, 32)
        assert isinstance(actual_compressor, zarr_BloscCodec)
        assert actual_compressor.cname == BloscCname.blosclz
        assert actual_compressor.clevel == 1

    def test_user_provided_client(self, mdio_dataset_path: str) -> None:
        """Test when user provides a dask client is present."""
        conf = OptimizedAccessPatternConfig(
            optimize_dimensions={"time": (128, 128, 4)},
            processing_chunks={"inline": 128, "crossline": 128, "time": 128},
        )
        ds = open_mdio(mdio_dataset_path)

        with Client(processes=False):
            optimize_access_patterns(ds, conf)

    def test_missing_default_variable_name(self, mdio_dataset_path: str) -> None:
        """Test case where default variable name is missing from dataset attributes."""
        conf = OptimizedAccessPatternConfig(
            optimize_dimensions={"time": (128, 128, 4)},
            processing_chunks={"inline": 128, "crossline": 128, "time": 128},
        )
        ds = open_mdio(mdio_dataset_path)
        del ds.attrs["attributes"]

        with pytest.raises(ValueError, match="Default variable name is missing from dataset attributes"):
            optimize_access_patterns(ds, conf)

    def test_missing_stats(self, mdio_dataset_path: str) -> None:
        """Test case where statistics are missing from default variable."""
        conf = OptimizedAccessPatternConfig(
            optimize_dimensions={"time": (128, 128, 4)},
            processing_chunks={"inline": 128, "crossline": 128, "time": 128},
        )
        ds = open_mdio(mdio_dataset_path)
        del ds["amplitude"].attrs["statsV1"]

        with pytest.raises(ValueError, match="Statistics are missing from data"):
            optimize_access_patterns(ds, conf)

    def test_invalid_optimize_access_patterns(self, mdio_dataset_path: str) -> None:
        """Test when optimize_dimensions contains invalid dimensions."""
        conf = OptimizedAccessPatternConfig(
            optimize_dimensions={"time": (128, 128, 4), "invalid": (4, 2, 44)},
            processing_chunks={"inline": 128, "crossline": 128, "time": 128},
        )
        ds = open_mdio(mdio_dataset_path)

        with pytest.raises(ValueError, match="Dimension to optimize 'invalid' not found"):
            optimize_access_patterns(ds, conf)
