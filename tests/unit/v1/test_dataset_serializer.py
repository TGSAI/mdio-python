"""Tests the schema v1 dataset_serializer public API."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from zarr.codecs import ZFPY as zarr_ZFPY  # noqa: N811
from zarr.codecs import BloscCodec

from mdio import to_mdio
from mdio.builder.dataset_builder import MDIODatasetBuilder
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import ZFP as MDIO_ZFP
from mdio.builder.schemas.compressors import Blosc as mdio_Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.compressors import BloscShuffle
from mdio.builder.schemas.compressors import ZFPMode as mdio_ZFPMode
from mdio.builder.schemas.dimension import NamedDimension
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredField
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.v1.dataset import Dataset
from mdio.builder.schemas.v1.dataset import DatasetMetadata
from mdio.builder.schemas.v1.variable import Coordinate
from mdio.builder.schemas.v1.variable import Variable
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.builder.xarray_builder import _compressor_to_encoding
from mdio.builder.xarray_builder import _get_all_named_dimensions
from mdio.builder.xarray_builder import _get_coord_names
from mdio.builder.xarray_builder import _get_dimension_names
from mdio.builder.xarray_builder import _get_fill_value
from mdio.builder.xarray_builder import _get_zarr_chunks
from mdio.builder.xarray_builder import _get_zarr_shape
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.constants import fill_value_map

from .helpers import make_seismic_poststack_3d_acceptance_dataset

if TYPE_CHECKING:
    from pathlib import Path

try:  # pragma: no cover
    import zfpy
except ImportError:
    zfpy = None


def test_get_all_named_dimensions() -> None:
    """Test _get_all_named_dimensions function."""
    dim1 = NamedDimension(name="inline", size=100)
    dim2 = NamedDimension(name="crossline", size=200)
    dim3 = NamedDimension(name="depth", size=300)
    v1 = Variable(name="named_dims", data_type=ScalarType.FLOAT32, dimensions=[dim1, dim2, dim3])
    v2 = Variable(
        name="string_dims",
        data_type=ScalarType.FLOAT32,
        dimensions=["inline", "crossline", "depth"],
    )
    v3 = Variable(name="unresolved_dims", data_type=ScalarType.FLOAT32, dimensions=["x", "y", "z"])
    ds = Dataset(
        variables=[v1, v2, v3],
        metadata=DatasetMetadata(name="test_dataset", api_version="1.0.0", created_on="2023-10-01T00:00:00Z"),
    )

    all_dims = _get_all_named_dimensions(ds)
    # Only 3 named dimensions could be resolved.
    # The dimension names "x", "y', "z" are unresolvable.
    assert set(all_dims) == {"inline", "crossline", "depth"}


def test_get_dimension_names() -> None:
    """Test _get_dimension_names function with various dimension types."""
    dim1 = NamedDimension(name="inline", size=100)
    dim2 = NamedDimension(name="crossline", size=200)

    # Test case 1: Variable with NamedDimension
    var_named_dims = Variable(
        name="Variable with NamedDimension dimensions",
        data_type=ScalarType.FLOAT32,
        dimensions=[dim1, dim2],
    )
    assert set(_get_dimension_names(var_named_dims)) == {"inline", "crossline"}

    # Test case 2: Variable with string dimensions
    var_string_dims = Variable(
        name="Variable with string dimensions",
        data_type=ScalarType.FLOAT32,
        dimensions=["x", "y", "z"],
    )
    assert set(_get_dimension_names(var_string_dims)) == {"x", "y", "z"}

    # Test case 3: Mixed NamedDimension and string dimensions
    # NOTE: mixing NamedDimension and string dimensions is not allowed by the Variable schema


def test_get_coord_names() -> None:
    """Comprehensive test for _get_coord_names function covering all scenarios."""
    dim1 = NamedDimension(name="inline", size=100)
    dim2 = NamedDimension(name="crossline", size=200)

    # Test 1: Variable with Coordinate objects
    coord1 = Coordinate(name="x_coord", dimensions=[dim1, dim2], data_type=ScalarType.FLOAT32)
    coord2 = Coordinate(name="y_coord", dimensions=[dim1, dim2], data_type=ScalarType.FLOAT64)
    variable_coords = Variable(
        name="Variable with Coordinate objects",
        data_type=ScalarType.FLOAT32,
        dimensions=[dim1, dim2],
        coordinates=[coord1, coord2],
    )
    assert set(_get_coord_names(variable_coords)) == {"x_coord", "y_coord"}

    # Test 2: Variable with string coordinates
    variable_strings = Variable(
        name="Variable with string coordinates",
        data_type=ScalarType.FLOAT32,
        dimensions=[dim1, dim2],
        coordinates=["lat", "lon", "time"],
    )
    assert set(_get_coord_names(variable_strings)) == {"lat", "lon", "time"}

    # Test 3: Variable with mixed coordinate types
    # NOTE: mixing Coordinate objects and coordinate name strings is not allowed by the
    # Variable schema


def test_get_zarr_shape() -> None:
    """Test for _get_zarr_shape function."""
    d1 = NamedDimension(name="inline", size=100)
    d2 = NamedDimension(name="crossline", size=200)
    d3 = NamedDimension(name="depth", size=300)
    all_named_dims = {"inline": d1, "crossline": d2, "depth": d3}
    v1 = Variable(name="named dims var", data_type=ScalarType.FLOAT32, dimensions=[d1, d2, d3])
    v2 = Variable(name="str var", data_type=ScalarType.FLOAT32, dimensions=["inline", "crossline", "depth"])
    Dataset(
        variables=[v1, v2],
        metadata=DatasetMetadata(name="test_dataset", api_version="1.0.0", created_on="2023-10-01T00:00:00Z"),
    )

    assert _get_zarr_shape(v1, all_named_dims) == (100, 200, 300)
    assert _get_zarr_shape(v2, all_named_dims) == (100, 200, 300)


def test_get_zarr_chunks() -> None:
    """Test for _get_zarr_chunks function."""
    d1 = NamedDimension(name="inline", size=100)
    d2 = NamedDimension(name="crossline", size=200)
    d3 = NamedDimension(name="crossline", size=300)

    # Test 1: Variable with chunk defined in metadata
    chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=(10, 20, 30)))
    metadata = VariableMetadata(chunk_grid=chunk_grid)
    v = Variable(name="seismic 3d var", data_type=ScalarType.FLOAT32, dimensions=[d1, d2], metadata=metadata)
    assert _get_zarr_chunks(v, all_named_dims=[d1, d2, d3]) == (10, 20, 30)

    # Test 2: Variable with no chunks defined
    v = Variable(name="seismic 3d var", data_type=ScalarType.FLOAT32, dimensions=[d1, d2, d3])
    assert _get_zarr_chunks(v, all_named_dims=[d1, d2, d3]) == (100, 200, 300)


def test_get_fill_value() -> None:
    """Test for _get_fill_value function."""
    # Test 1: ScalarType cases - should return values from fill_value_map
    scalar_types = [
        ScalarType.BOOL,
    ]
    for scalar_type in scalar_types:
        assert _get_fill_value(scalar_type) is None

    scalar_types = [
        ScalarType.FLOAT16,
        ScalarType.FLOAT32,
        ScalarType.FLOAT64,
    ]
    for scalar_type in scalar_types:
        assert np.isnan(_get_fill_value(scalar_type))

    scalar_types = [
        ScalarType.UINT8,
        ScalarType.UINT16,
        ScalarType.UINT32,
        ScalarType.INT8,
        ScalarType.INT16,
        ScalarType.INT32,
    ]
    for scalar_type in scalar_types:
        fill_value = _get_fill_value(scalar_type)
        assert fill_value_map[scalar_type] == fill_value

    scalar_types = [
        ScalarType.COMPLEX64,
        ScalarType.COMPLEX128,
        ScalarType.COMPLEX256,
    ]
    for scalar_type in scalar_types:
        val = _get_fill_value(scalar_type)
        assert isinstance(val, complex)
        assert np.isnan(val.real)
        assert np.isnan(val.imag)

    # Test 2: StructuredType
    f1 = StructuredField(name="cdp_x", format=ScalarType.INT32)
    f2 = StructuredField(name="cdp_y", format=ScalarType.INT32)
    f3 = StructuredField(name="elevation", format=ScalarType.FLOAT16)
    f4 = StructuredField(name="some_scalar", format=ScalarType.FLOAT16)
    structured_type = StructuredType(fields=[f1, f2, f3, f4])

    expected = np.array(
        (0, 0, 0.0, 0.0),
        dtype=np.dtype([("cdp_x", "<i4"), ("cdp_y", "<i4"), ("elevation", "<f2"), ("some_scalar", "<f2")]),
    )
    result = _get_fill_value(structured_type)
    assert expected == result

    # Test 3: String type - should return empty string
    result_string = _get_fill_value("string_type")
    assert result_string == ""

    # Test 4: Unknown type - should return None
    result_none = _get_fill_value(42)  # Invalid type
    assert result_none is None

    # Test 5: None input - should return None
    result_none_input = _get_fill_value(None)
    assert result_none_input is None


class TestCompressorToEncoding:
    """Test _compressor_to_encoding function for various configurations."""

    def test_compressor_encoding_blosc(self) -> None:
        """Blosc Compressor - should return zarr codec BloscCodec."""
        mdio_compressor = mdio_Blosc(cname=BloscCname.lz4, clevel=5, shuffle=BloscShuffle.bitshuffle, blocksize=1024)
        result = _compressor_to_encoding(mdio_compressor)

        assert isinstance(result["compressors"], BloscCodec)
        assert result["compressors"].cname == BloscCname.lz4
        assert result["compressors"].clevel == 5
        assert result["compressors"].shuffle == BloscShuffle.bitshuffle
        assert result["compressors"].blocksize == 1024

    def test_compressor_encoding_zfp(self) -> None:
        """ZFP Compressor - should return zarr codec ZFPY."""
        zfp_compressor = MDIO_ZFP(mode=mdio_ZFPMode.FIXED_RATE, tolerance=0.01, rate=8.0, precision=16)

        result_zfp = _compressor_to_encoding(zfp_compressor)
        assert result_zfp["compressors"] is None
        assert isinstance(result_zfp["serializer"], zarr_ZFPY)
        assert result_zfp["serializer"].codec_config["mode"] == 2  # fixed rate
        assert result_zfp["serializer"].codec_config["tolerance"] == 0.01
        assert result_zfp["serializer"].codec_config["rate"] == 8.0
        assert result_zfp["serializer"].codec_config["precision"] == 16

    def test_compressor_encoding_zfp_missing(self) -> None:
        """ZFP Compressor - should raise ImportError if zfpy is not installed."""
        zfp_compressor = MDIO_ZFP(mode=mdio_ZFPMode.FIXED_RATE, tolerance=0.01, rate=8.0, precision=16)

        with patch("mdio.builder.xarray_builder._import_numcodecs_zfpy") as mock_import:
            mock_import.side_effect = ImportError  # Simulate import failure

            with pytest.raises(ImportError, match="The 'zfpy' package is required for lossy compression."):
                _compressor_to_encoding(zfp_compressor)

    def test_compressor_encoding_none(self) -> None:
        """Test None encoding. Should return None."""
        result_none = _compressor_to_encoding(None)
        assert result_none is None

    def test_compressor_encoding_unsupported(self) -> None:
        """Test unsupported compressor type. Should raise TypeError."""
        unsupported_compressor = "invalid_compressor"
        with pytest.raises(TypeError, match="Unsupported compressor model"):
            _compressor_to_encoding(unsupported_compressor)


def test_to_xarray_dataset(tmp_path: Path) -> None:
    """Test building a complete dataset."""
    dataset = (
        MDIODatasetBuilder("test_dataset")
        .add_dimension("inline", 100)
        .add_dimension("crossline", 200)
        .add_dimension("depth", 300)
        .add_coordinate("inline", dimensions=("inline",), data_type=ScalarType.FLOAT64)
        .add_coordinate("crossline", dimensions=("crossline",), data_type=ScalarType.FLOAT64)
        .add_coordinate("x_coord", dimensions=("inline", "crossline"), data_type=ScalarType.FLOAT32)
        .add_coordinate("y_coord", dimensions=("inline", "crossline"), data_type=ScalarType.FLOAT32)
        .add_variable(
            "data",
            long_name="Test Data",
            dimensions=("inline", "crossline", "depth"),
            coordinates=("inline", "crossline", "x_coord", "y_coord"),
            data_type=ScalarType.FLOAT32,
        )
        .build()
    )

    xr_ds = to_xarray_dataset(dataset)

    file_path = f"{tmp_path}/{xr_ds.attrs['name']}.zarr"
    to_mdio(dataset=xr_ds, output_path=file_path, mode="w", compute=False)


def test_seismic_poststack_3d_acceptance_to_xarray_dataset(tmp_path: Path) -> None:
    """Test building a complete dataset."""
    dataset = make_seismic_poststack_3d_acceptance_dataset("Seismic")

    xr_ds = to_xarray_dataset(dataset)

    file_path = f"{tmp_path}/{xr_ds.attrs['name']}.zarr"
    to_mdio(xr_ds, output_path=file_path, mode="w-", compute=False)
