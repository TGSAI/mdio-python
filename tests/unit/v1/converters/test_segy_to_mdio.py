"""Tests for segy-to_mdio convenience functions."""

import numpy as np
import pytest
import xarray as xr
from segy.arrays import HeaderArray as SegyHeaderArray

from mdio.converters.segy import _populate_non_dim_coordinates
from mdio.core.dimension import Dimension
from mdio.core.grid import Grid


def test__populate_non_dim_coordinates() -> None:
    """Test population of non-dimensional coordinates."""
    dim_in, dim_xl, dim_of, dim_az, dim_tm = 2, 3, 4, 5, 1
    il = Dimension(name="inline", coords=np.arange(dim_in))
    xl = Dimension(name="crossline", coords=np.arange(dim_xl))
    of = Dimension(name="offset", coords=np.arange(dim_of))
    az = Dimension(name="azimuth", coords=np.arange(dim_az))
    tm = Dimension(name="time", coords=np.arange(dim_tm))

    r = np.random.rand(dim_in, dim_xl, dim_of, dim_az)

    il_, xl_, of_, az_ = np.meshgrid(il.coords, xl.coords, of.coords, az.coords, indexing="ij")
    diff = 1000 * il_ + 100 * xl_ + 10 * of_ + 1 * az_  # Different values for the same (il, xl)
    same = 1000 * il_ + 100 * xl_  # Same values for the same (il, xl)
    near = 1000 * il_ + 100 * xl_ + 1e-09 * r  # Near same values for the same (il, xl)
    # NOTE: near[1][1][1][1]: np.float64(1100.0000000003308)

    data_type = [
        ("inline", "<i4"),
        ("crossline", "<i4"),
        ("offset", "<i4"),
        ("azimuth", "<i4"),
        ("cdp_diff", "<f8"),
        ("cdp_same", "<f8"),
        ("cdp_near", "<f8"),
    ]
    data_list = [
        (i, j, k, m, diff[i, j, k, m], same[i, j, k, m], near[i, j, k, m])
        for i in range(dim_in)
        for j in range(dim_xl)
        for k in range(dim_of)
        for m in range(dim_az)
    ]
    segy_headers = SegyHeaderArray(np.array(data_list, dtype=data_type))

    grid = Grid(dims=[il, xl, of, az, tm])
    grid.build_map(segy_headers)

    ds = xr.Dataset(
        data_vars={
            "amplitude": (
                ["inline", "crossline", "offset", "azimuth", "time"],
                np.zeros((dim_in, dim_xl, dim_of, dim_az, dim_tm), dtype=np.float32),
            ),
        },
        coords={
            # Define coordinates with their dimensions and values
            "inline": il.coords,
            "crossline": xl.coords,
            "offset": of.coords,
            "azimuth": az.coords,
            "time": tm.coords,
            "cdp_diff": (["inline", "crossline"], np.zeros((dim_in, dim_xl), dtype=np.float64)),
            "cdp_same": (["inline", "crossline"], np.zeros((dim_in, dim_xl), dtype=np.float64)),
            "cdp_near": (["inline", "crossline"], np.zeros((dim_in, dim_xl), dtype=np.float64)),
        },
    )

    # "cdp_diff" has different values for the same (il, xl)
    coordinate_headers: dict[str, SegyHeaderArray] = {
        "cdp_diff": segy_headers["cdp_diff"],
    }
    expected_err = "Coordinate 'cdp_diff' has non-identical values along reduced dimensions."
    with pytest.raises(ValueError, match=expected_err):
        ds_populated, _ = _populate_non_dim_coordinates(ds, grid, coordinate_headers, [])

    # "cdp_same" has identical values for the same (il, xl)
    # "cdp_near" has near identical values for the same (il, xl)
    coordinate_headers: dict[str, SegyHeaderArray] = {
        "cdp_same": segy_headers["cdp_same"],
        "cdp_near": segy_headers["cdp_near"],
    }
    ds_populated, _ = _populate_non_dim_coordinates(ds, grid, coordinate_headers, [])
    expected_values = np.array([[0.0, 100.0, 200.0], [1000.0, 1100.0, 1200.0]], dtype=np.float32)
    assert np.allclose(ds_populated["cdp_same"].values, expected_values)
    assert np.allclose(ds_populated["cdp_near"].values, expected_values)
    # NOTE: ds_populated['cdp_near'].values[1][1]: np.float64(1100.0000000008847)
