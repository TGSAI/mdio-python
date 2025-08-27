import math
import random
import time
import numpy as np
import xarray as xr

from segy.arrays import HeaderArray as SegyHeaderArray

from mdio.converters.segy import _populate_non_dim_coordinates
from mdio.core.dimension import Dimension
from mdio.core.grid import Grid

def test__populate_non_dim_coordinates() ->None:

    dim_in, dim_xl, dim_of, dim_az, dim_tm = 2, 3, 4, 5, 1
    il = Dimension(name="inline", coords=np.arange(dim_in))
    xl = Dimension(name="crossline", coords=np.arange(dim_xl))
    of = Dimension(name="offset", coords=np.arange(dim_of))
    az = Dimension(name="azimuth", coords=np.arange(dim_az))
    tm = Dimension(name="time", coords=np.arange(dim_tm))
    
    r = np.random.rand(dim_in, dim_xl, dim_of, dim_az)

    il_, xl_, of_, az_ = np.meshgrid(il.coords, xl.coords, of.coords, az.coords, indexing='ij')
    diff = 1000*il_ + 100*xl_ + 10*of_ + 1*az_   # Different values for the same (il, xl)
    same = 1000*il_ + 100*xl_                    # Same values for the same (il, xl)
    near = 1000*il_ + 100*xl_ + 1e-09 * r        # Near same values for the same (il, xl)
    # NOTE: near[1][1][1][1]: np.float64(1100.0000000003308)

    data_type=[('inline', '<i4'), ('crossline', '<i4'), ('offset', '<i4'), ('azimuth', '<i4'),
               ('cdp_x', '<f8'), ('cdp_y', '<f8'), ('cdp_z', '<f8'),]
    data_list = []
    for i in range(dim_in):
        for j in range(dim_xl):
            for k in range(dim_of):
                for l in range(dim_az):
                    data_list.append((i, j, k, l, diff[i,j,k,l], same[i,j,k,l], near[i,j,k,l]))
    segy_headers = SegyHeaderArray(np.array(data_list, dtype=data_type))

    grid = Grid(dims=[il, xl, of, az, tm])
    grid.build_map(segy_headers)

    ds = xr.Dataset(
        data_vars={
            'amplitude': (["inline", "crossline", "offset", "azimuth", "time"],
                          np.zeros((dim_in, dim_xl, dim_of, dim_az, dim_tm), dtype=np.float32)),
        },
        coords={
            # Define coordinates with their dimensions and values
            'inline': il.coords,
            'crossline': xl.coords,
            'offset': of.coords,
            'azimuth': az.coords,
            'time': tm.coords,
            'cdp_x': (["inline", "crossline"], np.zeros((dim_in, dim_xl), dtype=np.float64)),
            'cdp_y': (["inline", "crossline"], np.zeros((dim_in, dim_xl), dtype=np.float64)),
            'cdp_z': (["inline", "crossline"], np.zeros((dim_in, dim_xl), dtype=np.float64))
        }
    )

    coordinate_headers: dict[str, SegyHeaderArray] = {
        'cdp_x': segy_headers['cdp_x'],
        'cdp_y': segy_headers['cdp_y'],
        'cdp_z': segy_headers['cdp_z']
    }

    is_good, ds_populated, _ = _populate_non_dim_coordinates(ds, grid, coordinate_headers, [])

    assert is_good['cdp_x'] is False
    assert is_good['cdp_y'] is True
    assert is_good['cdp_z'] is True
    expected_values = np.array([[0., 100., 200.], [1000., 1100., 1200.]], dtype=np.float32)
    assert np.allclose(ds_populated['cdp_x'].values, expected_values)
    assert np.allclose(ds_populated['cdp_y'].values, expected_values)
    assert np.allclose(ds_populated['cdp_z'].values, expected_values)
    # NOTE: ds_populated['cdp_z'].values[1][1]: np.float64(1100.0000000008847)

    pass
