"""End to end testing for SEG-Y to MDIO conversion v1."""

from __future__ import annotations

import numcodecs
import numpy as np
import pytest
import xarray as xr
import zarr
from segy.standards import get_segy_standard
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding

from mdio.converters.segy import segy_to_mdio
from mdio.converters.type_converter import to_numpy_dtype
from mdio.core.storage_location import StorageLocation
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredField
from mdio.schemas.dtype import StructuredType
from mdio.schemas.v1.templates.template_registry import TemplateRegistry


def _slice_three_values(dims: tuple[int], values_from_start: bool) -> tuple[slice, ...]:
    if values_from_start:
        slices = tuple([slice(0, 3) for _ in range(len(dims))])
    else:
        slices = tuple([slice(-3, None) for _ in range(len(dims))])
    return slices


def _get_actual_value(arr: xr.DataArray) -> np.ndarray:
    return arr.values[_slice_three_values(arr.shape, values_from_start=True)]


def _validate_variable(  # noqa PLR0913
    dataset: xr.Dataset,
    name: str,
    shape: list[int],
    dims: list[str],
    data_type: np.dtype,  # noqa ARG001
    # expected_values: range | None,
    # actual_func: Callable,
    expected_values: np.ndarray,
    actual_func: callable[[xr.DataArray], np.ndarray],
) -> None:
    arr = dataset[name]
    assert shape == arr.shape
    assert set(dims) == set(arr.dims)
    # assert data_type == arr.dtype
    actual_values = actual_func(arr)
    assert np.array_equal(expected_values, actual_values)


def test_segy_to_mdio_v1__f3() -> None:
    """Test the SEG-Y to MDIO conversion for the f3 equinor/segyio dataset."""
    # The f3 dataset comes from
    # equinor/segyio (https://github.com/equinor/segyio) project (GNU LGPL license)
    # wget https://github.com/equinor/segyio/blob/main/test-data/f3.sgy

    pref_path = "/DATA/equinor-segyio/f3.sgy"
    mdio_path = f"{pref_path}_mdio_v1"

    segy_sec = get_segy_standard(1.0)
    segy_to_mdio(
        segy_spec=segy_sec,
        mdio_template=TemplateRegistry().get("PostStack3DTime"),
        input_location=StorageLocation(pref_path),
        output_location=StorageLocation(mdio_path),
        overwrite=True,
    )

    # Load Xarray dataset from the MDIO file
    ds = xr.open_dataset(mdio_path, engine="zarr")

    # Tests "inline" variable
    expected = np.array([111, 112, 113])
    _validate_variable(ds, "inline", (23,), ["inline"], np.int32, expected, _get_actual_value)

    # Tests "crossline" variable
    expected = np.array([875, 876, 877])
    _validate_variable(ds, "crossline", (18,), ["crossline"], np.int32, expected, _get_actual_value)

    # Tests "time" variable
    expected = np.array([0, 4, 8])
    _validate_variable(ds, "time", (75,), ["time"], np.int64, expected, _get_actual_value)

    # Tests "cdp_x" variable
    expected = np.array(
        [[6201972, 6202222, 6202472], [6201965, 6202215, 6202465], [6201958, 6202208, 6202458]]
    )
    _validate_variable(
        ds, "cdp_x", (23, 18), ["inline", "crossline"], np.int32, expected, _get_actual_value
    )

    # Tests "cdp_y" variable
    expected = np.array(
        [
            [60742329, 60742336, 60742343],
            [60742579, 60742586, 60742593],
            [60742828, 60742835, 60742842],
        ]
    )
    _validate_variable(
        ds, "cdp_y", (23, 18), ["inline", "crossline"], np.int32, expected, _get_actual_value
    )

    # Tests "headers" variable
    data_type = segy_sec.trace.header.dtype
    expected = np.array(
        [
            [6201972, 6202222, 6202472],
            [6201965, 6202215, 6202465],
            [6201958, 6202208, 6202458],
        ],
        dtype=np.int32,
    )

    def get_actual_headers(arr: xr.DataArray) -> np.ndarray:
        cdp_x_headers = arr.values["cdp_x"]
        return cdp_x_headers[_slice_three_values(arr.shape, values_from_start=True)]

    _validate_variable(
        ds, "headers", (23, 18), ["inline", "crossline"], data_type, expected, get_actual_headers
    )

    # Tests "trace_mask" variable
    expected = np.array([[True, True, True], [True, True, True], [True, True, True]])
    _validate_variable(
        ds, "trace_mask", (23, 18), ["inline", "crossline"], np.bool, expected, _get_actual_value
    )

    # Tests "amplitude" variable
    expected = np.array(
        [
            [[487.0, -1104.0, -1456.0], [-129.0, -1728.0, 445.0], [-1443.0, 741.0, 1458.0]],
            [[2464.0, 3220.0, 1362.0], [686.0, 530.0, -282.0], [3599.0, 2486.0, 433.0]],
            [[4018.0, 5159.0, 2087.0], [-81.0, -3039.0, -1850.0], [2898.0, 1060.0, -121.0]],
        ]
    )

    def get_actual_amplitudes(arr: xr.DataArray) -> np.ndarray:
        return arr.values[_slice_three_values(arr.shape, values_from_start=False)]

    _validate_variable(
        ds,
        "amplitude",
        (23, 18, 75),
        ["inline", "crossline", "time"],
        np.float32,
        expected,
        get_actual_amplitudes,
    )


@pytest.mark.skip(reason="Bug reproducer for the issue 582")
def test_bug_reproducer_structured_xr_to_zar() -> None:
    """Bug reproducer for the issue https://github.com/TGSAI/mdio-python/issues/582.

    Will be removed in the when the bug is fixed
    """
    shape = (4, 4, 2)
    dim_names = ["inline", "crossline", "depth"]
    chunks = (2, 2, 2)
    # Pretend that we created a pydantic model from a template
    structured_type = StructuredType(
        fields=[
            StructuredField(name="cdp_x", format=ScalarType.INT32),
            StructuredField(name="cdp_y", format=ScalarType.INT32),
            StructuredField(name="elevation", format=ScalarType.FLOAT16),
            StructuredField(name="some_scalar", format=ScalarType.FLOAT16),
        ]
    )

    xr_dataset = xr.Dataset()

    # Add traces to the dataset, shape = (4, 4, 2) of floats
    traces_zarr = zarr.zeros(shape=shape, dtype=np.float32, zarr_format=2)
    traces_xr = xr.DataArray(traces_zarr, dims=dim_names)
    traces_xr.encoding = {
        "_FillValue": np.nan,
        "chunks": chunks,
        "chunk_key_encoding": V2ChunkKeyEncoding(separator="/").to_dict(),
        "compressor": numcodecs.Blosc(cname="zstd", clevel=5, shuffle=1, blocksize=0),
    }
    xr_dataset["traces"] = traces_xr

    # Add headers to the dataset, shape = (4, 4) of structured type
    data_type = to_numpy_dtype(structured_type)

    # Validate the conversion
    assert data_type == np.dtype(
        [("cdp_x", "<i4"), ("cdp_y", "<i4"), ("elevation", "<f2"), ("some_scalar", "<f2")]
    )
    fill_value = np.zeros((), dtype=data_type)
    headers_zarr = zarr.zeros(shape=shape[:-1], dtype=data_type, zarr_format=2)
    headers_xr = xr.DataArray(headers_zarr, dims=dim_names[:-1])
    headers_xr.encoding = {
        "_FillValue": fill_value,
        "chunks": chunks[:-1],
        "chunk_key_encoding": V2ChunkKeyEncoding(separator="/").to_dict(),
        "compressor": numcodecs.Blosc(cname="zstd", clevel=5, shuffle=1, blocksize=0),
    }
    xr_dataset["headers"] = headers_xr

    # See _populate_dims_coords_and_write_to_zarr()
    # The compute=True because we would also write to Zarr the coord values here
    xr_dataset.to_zarr(
        store="/tmp/reproducer_xr.zarr",  # noqa: S108
        mode="w",
        write_empty_chunks=False,
        zarr_format=2,
        compute=True,
    )

    # In _populate_trace_mask_and_write_to_zarr
    # We do another write of "trace_mask" to the same Zarr store and remove it
    # from the dataset

    # ----------------------------------------------
    # Now will will do parallel write of the data and the headers
    # see blocked_io.to_zarr -> trace_worker

    not_null = np.array(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ]
    )
    hdr = (11, 22, -33.0, 44.0)
    headers = np.array([hdr, hdr, hdr, hdr], dtype=data_type)
    trace = np.array(
        [[100.0, 200.0], [300.0, 400.0], [500.0, 600.0], [700.0, 800.0]], dtype=np.float32
    )

    # Here is one iteration of it:
    ds_to_write = xr_dataset[["traces", "headers"]]
    # We do not have any coords to reset
    # ds_to_write = ds_to_write.reset_coords()

    ds_to_write["headers"].data[not_null] = headers
    ds_to_write["headers"].data[~not_null] = 0
    ds_to_write["traces"].data[not_null] = trace

    region = {
        "inline": slice(0, 2, None),
        "crossline": slice(0, 2, None),
        "depth": slice(0, 2, None),
    }

    sub_dataset = ds_to_write.isel(region)
    sub_dataset.to_zarr(
        store="/tmp/reproducer_xr.zarr",  # noqa: S108
        region=region,
        mode="r+",
        write_empty_chunks=False,
        zarr_format=2,
    )
