"""Utils for reading MDIO dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import zarr
from upath import UPath
from xarray import Dataset as xr_Dataset
from xarray import open_zarr as xr_open_zarr
from xarray.backends.writers import to_zarr as xr_to_zarr

from mdio.constants import ZarrFormat
from mdio.core.zarr_io import zarr_warnings_suppress_unstable_structs_v3

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from xarray import Dataset
    from xarray.core.types import T_Chunks
    from xarray.core.types import ZarrWriteModes

def _normalize_path(path: UPath | Path | str) -> UPath:
    """Normalize a path to a UPath.

    For gs:// paths, the fake GCS server configuration is handled via storage_options
    in _normalize_storage_options().
    """
    from upath import UPath

    return UPath(path)

def _normalize_storage_options(path: UPath) -> dict[str, Any] | None:
    """Normalize and patch storage options for UPath paths.

    - Extracts any existing options from the UPath.
    - Automatically redirects gs:// URLs to a local fake-GCS endpoint
      when testing (localhost:4443).
    """
    import gcsfs

    # Start with any existing options from UPath
    storage_options = dict(path.storage_options) if len(path.storage_options) else {}

    # Redirect gs:// to local fake-GCS server for testing
    if str(path).startswith("gs://"):
        fs = gcsfs.GCSFileSystem(
            endpoint_url="http://localhost:4443",
            token="anon",
        )
        base_url = getattr(getattr(fs, "session", None), "_base_url", "http://localhost:4443")
        print(f"[mdio.utils] Redirecting GCS path to local fake server: {base_url}")
        storage_options["fs"] = fs

    return storage_options or None

# def _normalize_path(path: UPath | Path | str) -> UPath:
#     return UPath(path)


# def _normalize_storage_options(path: UPath) -> dict[str, Any] | None:
#     return None if len(path.storage_options) == 0 else path.storage_options


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
    import zarr

    input_path = _normalize_path(input_path)
    storage_options = _normalize_storage_options(input_path)
    zarr_format = zarr.config.get("default_zarr_format")

    return xr_open_zarr(
        input_path.as_posix(),
        chunks=chunks,
        storage_options=storage_options,
        mask_and_scale=zarr_format == ZarrFormat.V3,  # off for v2, on for v3
        consolidated=zarr_format == ZarrFormat.V2,  # on for v2, off for v3
    )

def to_mdio(
    dataset: Dataset,
    output_path: UPath | Path | str,
    mode: ZarrWriteModes | None = None,
    *,
    compute: bool = True,
    region: Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None = None,):
    """Write dataset contents to an MDIO output_path."""
    import gcsfs
    import zarr

    output_path = _normalize_path(output_path)
    zarr_format = zarr.config.get("default_zarr_format")

    # For GCS paths, create FSMap for fake GCS server
    if str(output_path).startswith("gs://"):
        fs = gcsfs.GCSFileSystem(
            endpoint_url="http://localhost:4443",
            token="anon",
        )
        base_url = getattr(getattr(fs, "session", None), "_base_url", "http://localhost:4443")
        print(f"[mdio.utils] Using fake GCS mapper via {base_url}")
        store = fs.get_mapper(output_path.as_posix().replace("gs://", ""))
        storage_options = None  # Must be None when passing a mapper
    else:
        store = output_path.as_posix()
        storage_options = _normalize_storage_options(output_path)

    print(f"[mdio.utils] Writing to store: {store}")
    print(f"[mdio.utils] Storage options: {storage_options}")

    kwargs = dict(
        dataset=dataset,
        store=store,
        mode=mode,
        compute=compute,
        consolidated=zarr_format == ZarrFormat.V2,
        region=region,
        write_empty_chunks=False,
    )
    if storage_options is not None and not isinstance(store, dict):
        kwargs["storage_options"] = storage_options

    with zarr_warnings_suppress_unstable_structs_v3():
        xr_to_zarr(**kwargs)


# def to_mdio(  # noqa: PLR0913
#     dataset: Dataset,
#     output_path: UPath | Path | str,
#     mode: ZarrWriteModes | None = None,
#     *,
#     compute: bool = True,
#     region: Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None = None,
# ) -> None:
#     """Write dataset contents to an MDIO output_path."""
#     import gcsfs, zarr

#     output_path = _normalize_path(output_path)

#     if output_path.as_posix().startswith("gs://"):
#         fs = gcsfs.GCSFileSystem(
#             endpoint_url="http://localhost:4443",
#             token="anon",
#         )

#         base_url = getattr(getattr(fs, "session", None), "_base_url", "http://localhost:4443")
#         print(f"Using custom fake GCS filesystem with endpoint {base_url}")

#         # Build a mapper so all I/O uses the fake GCS server
#         mapper = fs.get_mapper(output_path.as_posix().replace("gs://", ""))
#         store = mapper
#         storage_options = None  # must be None when passing a mapper
#     else:
#         store = output_path.as_posix()
#         storage_options = _normalize_storage_options(output_path) or {}

#     print(f"Writing to store: {store}")
#     zarr_format = zarr.config.get("default_zarr_format")

#     kwargs = dict(
#         dataset=dataset,
#         store=store,
#         mode=mode,
#         compute=compute,
#         consolidated=zarr_format == ZarrFormat.V2,
#         region=region,
#         write_empty_chunks=False,
#     )
#     if storage_options is not None:
#         kwargs["storage_options"] = storage_options

#     with zarr_warnings_suppress_unstable_structs_v3():
#         xr_to_zarr(**kwargs)




