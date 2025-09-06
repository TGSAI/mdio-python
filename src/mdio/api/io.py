"""Utils for reading MDIO dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from upath import UPath
from xarray import Dataset as xr_Dataset
from xarray import open_zarr as xr_open_zarr
from xarray.backends.api import to_zarr as xr_to_zarr

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from xarray import Dataset
    from xarray.core.types import T_Chunks
    from xarray.core.types import ZarrWriteModes


def _normalize_path(path: UPath | Path | str) -> UPath:
    return UPath(path)


def _normalize_storage_options(path: UPath) -> dict[str, Any] | None:
    return None if len(path.storage_options) == 0 else path.storage_options


def open_mdio(input_path: UPath | Path | str, chunks: T_Chunks = None) -> xr_Dataset:
    """Open a Zarr dataset from the specified universal file path.

    Args:
        input_path: Universal input path for the MDIO dataset.
        chunks: If provided, loads data into dask arrays with new chunking.
            - ``chunks="auto"`` will use dask ``auto`` chunking
            - ``chunks=None`` skips using dask, which is generally faster for small arrays.
            - ``chunks=-1`` loads the data with dask using a single chunk for all arrays.
            - ``chunks={}`` loads the data with dask using the engine's preferred chunk size (on disk).
            - ``chunks={dim: chunk, ...}`` loads the data with dask using the specified chunk size for each dimension.

            See dask chunking for more details.

    Returns:
        An Xarray dataset opened from the input path.
    """
    input_path = _normalize_path(input_path)
    storage_options = _normalize_storage_options(input_path)
    return xr_open_zarr(input_path.as_posix(), chunks=chunks, storage_options=storage_options)


def to_mdio(  # noqa: PLR0913
    dataset: Dataset,
    output_path: UPath | Path | str,
    mode: ZarrWriteModes | None = None,
    *,
    compute: bool = True,
    region: Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None = None,
    zarr_format: int = 3,
) -> None:
    """Write dataset contents to an MDIO output_path.

    Args:
        dataset: The dataset to write.
        output_path: The universal path of the output MDIO file.
        mode: Persistence mode: "w" means create (overwrite if exists)
            "w-" means create (fail if exists)
            "a" means override all existing variables including dimension coordinates (create if does not exist)
            "a-" means only append those variables that have ``append_dim``.
            "r+" means modify existing array *values* only (raise an error if any metadata or shapes would change).
            The default mode is "r+" if ``region`` is set and ``w-`` otherwise.
        compute: If True write array data immediately; otherwise return a ``dask.delayed.Delayed`` object that
            can be computed to write array data later. Metadata is always updated eagerly.
        region: Optional mapping from dimension names to either a) ``"auto"``, or b) integer slices, indicating
            the region of existing MDIO array(s) in which to write this dataset's data.
        zarr_format: The desired zarr format to target. The default is 3.
    """
    output_path = _normalize_path(output_path)
    storage_options = _normalize_storage_options(output_path)
    xr_to_zarr(
        dataset,
        store=output_path.as_posix(),  # xarray doesn't like URI when file:// is protocol
        mode=mode,
        compute=compute,
        consolidated=False,
        region=region,
        storage_options=storage_options,
        zarr_format=zarr_format,
        write_empty_chunks=False,
    )
