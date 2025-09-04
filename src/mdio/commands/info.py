"""MDIO Dataset information command."""

from __future__ import annotations

from typing import Any

import xarray as xr
from click import STRING
from click import argument
from click import command
from click import option
from click_params import JSON


@command(name="info")
@argument("mdio-path", type=STRING)
@option(
    "-storage",
    "--storage-options",
    required=False,
    help="Storage options for SEG-Y input file.",
    type=JSON,
)
def info(mdio_path: str, storage_options: dict[str, Any]) -> None:
    """Provide information on a MDIO dataset.

    By default, this returns human-readable information about the grid and stats for the dataset.
    If output-format is set to 'json' then a JSON is returned to facilitate parsing.
    """
    ds = xr.open_zarr(mdio_path, mask_and_scale=False, storage_options=storage_options)
    print(ds)


cli = info
