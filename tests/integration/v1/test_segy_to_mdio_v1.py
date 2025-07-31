from pyparsing import Any
import pytest
from segy.standards import get_segy_standard
import xarray as xr
import numpy as np
import zarr
import numcodecs

from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding

from mdio.converters.segy_to_mdio_v1 import segy_to_mdio_v1
from mdio.converters.segy_to_mdio_v1 import segy_to_mdio_v1_customized
from mdio.converters.segy_to_mdio_v1 import StorageLocation

from mdio.converters.type_converter import to_numpy_dtype
from mdio.schemas.dtype import ScalarType, StructuredField, StructuredType
from mdio.schemas.v1.dataset_serializer import _get_fill_value
from mdio.schemas.v1.templates.template_registry import TemplateRegistry

@pytest.mark.skip(reason="Need a test dataset to customize.")
def test_segy_to_mdio_v1_customized() -> None:
    """Test the custom SEG-Y to MDIO conversion."""
    pref_path = "/DATA/Teapot/filt_mig"
    mdio_path = f"{pref_path}_custom_v1.mdio"

    # NOTE: This test will fail because no indexs are provided for the required 'cdp_x' and 'cdp_y' fields
    # in the filt_mig dataset.
    index_bytes = (181, 185)
    index_names = ("inline", "crossline")
    index_types = ("int32", "int32")

    segy_to_mdio_v1_customized(
        segy_spec="1.0",
        mdio_template="PostStack3DTime",
        input=StorageLocation(f"{pref_path}.segy"),
        output=StorageLocation(mdio_path),
        index_bytes=index_bytes,
        index_names=index_names,
        index_types=index_types,
        overwrite=True
    )

    # Load Xarray dataset from the MDIO file
    dataset = xr.open_dataset(mdio_path, engine="zarr")
    pass

def slice_three_values(dims: tuple[int], values_from_start:bool):
    if values_from_start:
        slices = tuple([slice(0, 3) for _ in range(len(dims))])
    else:
        slices = tuple([slice(-3, None) for _ in range(len(dims))])
    return slices


def validate_variable(dataset: xr.Dataset,
                      name: str,
                      shape: list[int],
                      dims: list[str],
                      data_type: np.dtype,
                      expected_values: Any,
                      values_from_start: bool):
    d = dataset[name]
    assert shape == d.shape
    assert set(dims) == set(d.dims)
    assert data_type == d.dtype
    # Validate first/last values
    actual_values = d.values[slice_three_values(shape, values_from_start)]
    assert np.array_equal(expected_values, actual_values)


def test_segy_to_mdio_v1__f3() -> None:
    # The f3 dataset comes from
    # equinor/segyio (https://github.com/equinor/segyio) project (GNU LGPL license)
    # wget https://github.com/equinor/segyio/blob/main/test-data/f3.sgy

    pref_path = "/DATA/equinor-segyio/f3.sgy"
    mdio_path = f"{pref_path}_mdio_v1"

    segy_to_mdio_v1(
        segy_spec=get_segy_standard(1.0),
        mdio_template=TemplateRegistry().get("PostStack3DTime"),
        input=StorageLocation(pref_path),
        output=StorageLocation(mdio_path))

    # Load Xarray dataset from the MDIO file
    ds = xr.open_dataset(mdio_path, engine="zarr")

    data = np.array([111, 112, 113])
    validate_variable(ds, "inline", (23,), ["inline"], np.int32, data, True)

    data = np.array([875, 876, 877])
    validate_variable(ds, "crossline", (18,), [
                      "crossline"], np.int32, data, True)

    data = np.array([0, 4, 8])
    validate_variable(ds, "time", (75,), ["time"], np.int64, data, True)
    # assert (75,) == dataset["time"].shape

    data = np.array([[6201819, 6201826, 6201833],
                     [6201944, 6201951, 6201958],
                     [6202159, 6202166, 6202173]])
    validate_variable(ds, "cdp_x", (23, 18), [
                      "inline", "crossline"], np.int32, data, True)

    data = np.array([[60742329, 60742336, 60742343],
                     [60742579, 60742586, 60742593],
                     [60742828, 60742835, 60742842]])
    validate_variable(ds, "cdp_y", (23, 18), [
                      "inline", "crossline"], np.int32, data, True)

    data = np.array([[True, True, True],
                     [True, True, True],
                     [True, True, True]])
    validate_variable(ds, "trace_mask", (23, 18), [
                      "inline", "crossline"], np.bool, data, True)

    data = np.array([[[487., -1104., -1456.],
                      [-129., -1728.,   445.],
                      [-1443.,   741.,  1458.]],
                     [[2464.,  3220.,  1362.],
                      [686.,   530.,  -282.],
                      [3599.,  2486.,   433.]],
                     [[4018.,  5159.,  2087.],
                      [-81., -3039., -1850.],
                      [2898.,  1060.,  -121.]]])
    validate_variable(ds, "amplitude", (23, 18, 75), [
                      "inline", "crossline", "time"], np.float32, data, False)

@pytest.mark.skip(reason="Issue reproducer, not actually a test")
def test_repro_structured_xr_to_zar() ->None:
    """Reproducer for problems with the segy_to_mdio_v1 function.

    Will be removed in the when the final PR is submitted
    """
    shape = (4, 4, 2)
    dim_names = ['inline', 'crossline', 'depth']
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
        "compressor": numcodecs.Blosc(cname='zstd', clevel=5, shuffle=1, blocksize=0),
    }
    xr_dataset["traces"] = traces_xr

    # Add headers to the dataset, shape = (4, 4) of structured type
    data_type = to_numpy_dtype(structured_type)

    # Validate the conversion
    assert data_type == np.dtype([('cdp_x', '<i4'), ('cdp_y', '<i4'), ('elevation', '<f2'), ('some_scalar', '<f2')])
    fill_value = np.zeros((), dtype=data_type)
    headers_zarr = zarr.zeros(shape=shape[:-1], dtype=data_type, zarr_format=2)
    headers_xr = xr.DataArray(headers_zarr, dims=dim_names[:-1])
    headers_xr.encoding = {
        "_FillValue": fill_value,
        "chunks": chunks[:-1],
        "chunk_key_encoding": V2ChunkKeyEncoding(separator="/").to_dict(),
        "compressor": numcodecs.Blosc(cname='zstd', clevel=5, shuffle=1, blocksize=0),
    }
    xr_dataset["headers"] = headers_xr

    # See _populate_dims_coords_and_write_to_zarr()
    # The compute=True because we would also write to Zarr the coord values here
    xr_dataset.to_zarr(store="/tmp/reproducer_xr.zarr",
                        mode="w",
                        write_empty_chunks=False,
                        zarr_format=2,
                        compute=True)

    # In _populate_trace_mask_and_write_to_zarr
    # We do another write of "trace_mask" to the same Zarr store and remove it
    # from the dataset

    # ----------------------------------------------
    # Now will will do parallel write of the data and the headers
    # see blocked_io.to_zarr -> trace_worker

    not_null = np.array([[True, False, False, False],
                         [False, True, False, False],
                         [False, False, True, False],
                         [False, False, False, True]])
    hdr = (11, 22, -33.0, 44.0)
    headers = np.array([hdr,hdr,hdr,hdr], dtype=data_type)
    trace = np.array([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0], [700.0, 800.0]], dtype=np.float32)

    # Here is one iteration of it:
    ds_to_write = xr_dataset[["traces","headers"]]
    # We do not have any coords to reset
    # ds_to_write = ds_to_write.reset_coords()


    ds_to_write["headers"].data[not_null] = headers
    ds_to_write["headers"].data[~not_null] = 0
    ds_to_write["traces"].data[not_null] = trace

    region = {
        'inline': slice(0, 2, None),
        'crossline': slice(0, 2, None),
        'depth': slice(0, 2, None)
    }

    sub_dataset = ds_to_write.isel(region)
    sub_dataset.to_zarr(store="/tmp/reproducer_xr.zarr",
                        region=region,
                        mode="r+",
                        write_empty_chunks=False,
                        zarr_format=2)

    pass

# /home/vscode/.venv/lib/python3.13/site-packages/xarray/backends/zarr.py:945: in store
#     existing_vars, _, _ = conventions.decode_cf_variables(
# E - TypeError: Failed to decode variable 'headers': unhashable type: 'writeable void-scalar'

