from segy.standards import get_segy_standard
import xarray as xr
import numpy as np
import zarr
import numcodecs

from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding

from mdio.converters.segy_to_mdio_v1 import segy_to_mdio_v1
from mdio.converters.segy_to_mdio_v1_custom import StorageLocation

from mdio.converters.type_converter import to_numpy_dtype
from mdio.schemas.dtype import ScalarType, StructuredField, StructuredType
from mdio.schemas.v1.dataset_serializer import _get_fill_value
from mdio.schemas.v1.templates.template_registry import TemplateRegistry

def test_segy_to_mdio_v1() -> None:
    pref_path = "/DATA/export_masked/3d_stack"

    mdio_path = f"{pref_path}_v1.mdio"

    segy_to_mdio_v1(
        input= StorageLocation(f"{pref_path}.sgy"),
        output= StorageLocation(mdio_path),
        segy_spec= get_segy_standard(1.0),
        mdio_template= TemplateRegistry().get("PostStack3DTime"))


    # Load Xarray dataset from the MDIO file
    dataset = xr.open_dataset(mdio_path, engine="zarr")
    pass


def test_repro_structured_np_to_zar() ->None:
    """Reproducer for problems with the segy_to_mdio_v1 function.
    
    Will be removed in the when the final PR is submitted
    """

    # Create some sample data
    shape = (4, 4, 2)
    dim_names = ['x', 'y', 'z']
    chunks = (2, 2, 2)

    dtype = np.dtype([('cdp_x', '<i4'), ('cdp_y', '<i4'), ('elevation', '<f2'), ('some_scalar', '<f2')])

    # Create the DataArray
    np_data = xr.DataArray(
        data=np.zeros(shape, dtype=dtype),
        dims=dim_names,
        name='reproducer',
    )
    np_dataset = xr.Dataset()
    np_dataset["np_structured"] = np_data
    np_dataset.to_zarr(store="/tmp/reproducer_np.zarr",
                       mode="w",
                       write_empty_chunks=True,
                       zarr_format=2,
                       compute=True)

def test_repro_structured_xr_to_zar() ->None:
    """Reproducer for problems with the segy_to_mdio_v1 function.
    
    Will be removed in the when the final PR is submitted
    """
        
    shape = (4, 4, 2)
    dim_names = ['x', 'y', 'z']
    chunks = (2, 2, 2)

    structured_type = StructuredType(
        fields=[
                StructuredField(name="cdp_x", format=ScalarType.INT32),
                StructuredField(name="cdp_y", format=ScalarType.INT32),
                StructuredField(name="elevation", format=ScalarType.FLOAT16),
                StructuredField(name="some_scalar", format=ScalarType.FLOAT16),
        ]
    )
    dtype = to_numpy_dtype(structured_type)

    zarr_data = zarr.zeros(shape=shape[:-1], dtype=dtype, zarr_format=2)
    xr_array = xr.DataArray(zarr_data, dims=dim_names[:-1])
    chunk_key_encoding = V2ChunkKeyEncoding(separator="/").to_dict()
    encoding = {
        # NOTE: See Zarr documentation on use of fill_value and _FillValue in Zarr v2 vs v3
        "_FillValue": _get_fill_value(dtype),
        "chunks": chunks[:-1],
        "chunk_key_encoding": chunk_key_encoding,
        "compressor": numcodecs.Blosc(cname='zstd', clevel=5, shuffle=1, blocksize=0),
    }
    xr_array.encoding = encoding

    s = xr_array.shape
    c = xr_array.encoding.get("chunks")

    xr_dataset = xr.Dataset()
    xr_dataset["xr_structured"] = xr_array

    xr_dataset.to_zarr(store="/tmp/reproducer_xr.zarr",
                        mode="w",
                        write_empty_chunks=False,
                        zarr_format=2,
                        compute=True)

    # ----------------------------------------------


    not_null = np.array([[True, False, False, False],
                         [False, True, False, False],
                         [False, False, True, False],
                         [False, False, False, True]])

    v = (11, 22, -33.0, 44.0)
    traces = np.array([v,v,v,v], dtype=dtype)
    xr_dataset["xr_structured"].data[not_null] = traces
    
    xr_data = xr_dataset["xr_structured"]

    dim = shape[:-1]
    start_indices = tuple(
        dim * chunk for dim, chunk in zip(0, 4)
    )

    stop_indices = tuple(
        (dim + 1) * chunk for dim, chunk in zip(0, 4)
    )

    slices = tuple(
        slice(start, stop) for start, stop in zip(start_indices, stop_indices)
    )

    slices = dict(zip(dim_names[:-1], slices))

    # {'inline': slice(0, 20, None), 'crossline': slice(0, 20, None), 'time': slice(0, 201, None)}

    regions = tuple(slice(start, stop) for start, stop in zip(0, 4))

    region = tuple(regions[key] for key in dim_names[:-1])

    xr_dataset.to_zarr(store="/tmp/reproducer_xr.zarr",
                       region=region, 
                       mode="r+",
                       write_empty_chunks=False,
                       zarr_format=2,
                       compute=True)


    pass
