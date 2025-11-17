"""Integration tests for chunking module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask
import numpy as np
import pytest
from segy.factory import SegyFactory
from segy.schema import HeaderField
from segy.schema import ScalarType
from segy.standards import get_segy_standard

from mdio.api.io import open_mdio
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.template_registry import TemplateRegistry
from mdio.converters.segy import segy_to_mdio
from mdio.transpose_writers.chunking import from_variable

if TYPE_CHECKING:
    from pathlib import Path

    from segy.schema import SegySpec


dask.config.set(scheduler="synchronous")


@pytest.fixture
def synthetic_segy_spec() -> SegySpec:
    """SEG-Y specification for synthetic testing."""
    trace_header_fields = [
        HeaderField(name="inline", byte=189, format=ScalarType.INT32),
        HeaderField(name="crossline", byte=193, format=ScalarType.INT32),
        HeaderField(name="cdp_x", byte=181, format=ScalarType.INT32),
        HeaderField(name="cdp_y", byte=185, format=ScalarType.INT32),
    ]
    return get_segy_standard(1.0).customize(trace_header_fields=trace_header_fields)


@pytest.fixture
def synthetic_segy_file(fake_segy_tmp: Path, synthetic_segy_spec: SegySpec) -> Path:
    """Create a small synthetic 3D SEG-Y file."""
    segy_path = fake_segy_tmp / "synthetic_3d.sgy"
    inlines, crosslines, num_samples = np.arange(1, 9), np.arange(1, 17), 64
    num_traces = len(inlines) * len(crosslines)

    factory = SegyFactory(spec=synthetic_segy_spec, sample_interval=4000, samples_per_trace=num_samples)
    headers = factory.create_trace_header_template(num_traces)
    samples = factory.create_trace_sample_template(num_traces)

    trace_idx = 0
    for inline in inlines:
        for crossline in crosslines:
            headers["inline"][trace_idx] = inline
            headers["crossline"][trace_idx] = crossline
            headers["cdp_x"][trace_idx] = inline * 100
            headers["cdp_y"][trace_idx] = crossline * 100
            headers["coordinate_scalar"][trace_idx] = -100
            samples[trace_idx] = np.sin(np.linspace(0, 2 * np.pi * inline, num_samples)) * crossline
            trace_idx += 1

    with segy_path.open(mode="wb") as fp:
        fp.write(factory.create_textual_header())
        fp.write(factory.create_binary_header())
        fp.write(factory.create_traces(headers, samples))

    return segy_path


@pytest.fixture
def mdio_dataset(synthetic_segy_file: Path, synthetic_segy_spec: SegySpec, zarr_tmp: Path) -> Path:
    """Convert synthetic SEG-Y to MDIO."""
    mdio_path = zarr_tmp / "test_dataset.mdio"
    segy_to_mdio(
        segy_spec=synthetic_segy_spec,
        mdio_template=TemplateRegistry().get("PostStack3DTime"),
        input_path=synthetic_segy_file,
        output_path=mdio_path,
        overwrite=True,
    )
    return mdio_path


def test_single_variable_rechunk(mdio_dataset: Path) -> None:
    """Test creating a single rechunked variable."""
    chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=(2, 16, 64)))
    compressor = Blosc(cname="zstd", clevel=5, shuffle="shuffle")

    from_variable(
        dataset_path=mdio_dataset,
        source_variable="amplitude",
        new_variable="fast_inline",
        chunk_grid=chunk_grid,
        compressor=compressor,
        copy_metadata=True,
    )

    ds = open_mdio(mdio_dataset)

    # Verify new variable exists with correct chunks
    assert "fast_inline" in ds.data_vars
    assert ds["fast_inline"].encoding["chunks"] == (2, 16, 64)

    # Verify data integrity
    np.testing.assert_array_equal(ds["amplitude"].values, ds["fast_inline"].values)

    # Verify metadata copied
    assert ds["fast_inline"].attrs == ds["amplitude"].attrs


def test_multiple_variables_with_broadcasting(mdio_dataset: Path) -> None:
    """Test creating multiple variables with different settings and broadcasting."""
    new_variables = ["fast_inline", "fast_crossline", "fast_time"]
    chunk_grids = [
        RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=(2, 16, 64))),
        RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=(8, 4, 64))),
        RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=(8, 16, 16))),
    ]
    compressor = Blosc(cname="lz4", clevel=3, shuffle="shuffle")  # Single compressor broadcasts

    from_variable(
        dataset_path=mdio_dataset,
        source_variable="amplitude",
        new_variable=new_variables,
        chunk_grid=chunk_grids,
        compressor=compressor,
        copy_metadata=False,
    )

    ds = open_mdio(mdio_dataset)

    # Verify all variables created with correct chunks
    expected_chunks = [(2, 16, 64), (8, 4, 64), (8, 16, 16)]
    for var_name, chunks in zip(new_variables, expected_chunks, strict=True):
        assert var_name in ds.data_vars
        assert ds[var_name].encoding["chunks"] == chunks
        assert "compressors" in ds[var_name].encoding
        np.testing.assert_array_equal(ds["amplitude"].values, ds[var_name].values)

        # Metadata should not be copied
        assert len(ds[var_name].attrs) == 0 or ds[var_name].attrs != ds["amplitude"].attrs
