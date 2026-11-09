"""Reading, writing, and metadata updates for MDIO datasets."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import zarr
from upath import UPath
from xarray import Dataset as xr_Dataset
from xarray import open_zarr as xr_open_zarr
from xarray.backends.writers import to_zarr as xr_to_zarr
from zarr.codecs import BloscCodec
from zarr.codecs.numcodecs import Blosc

from mdio.constants import ZarrFormat
from mdio.core.zarr_io import zarr_warnings_suppress_unstable_structs_v3
from mdio.exceptions import MDIOMissingVariableError
from mdio.exceptions import MDIONotFoundError

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from xarray import Dataset
    from xarray.core.types import T_Chunks
    from xarray.core.types import ZarrWriteModes


def _normalize_path(path: UPath | Path | str) -> UPath:
    return UPath(path)


def _normalize_storage_options(path: UPath) -> dict[str, Any] | None:
    # UPath.storage_options returns a read-only mappingproxy which cannot be pickled. Copy it into a
    # plain dict so callers can safely pass it across process boundaries (e.g. spawned workers).
    return None if len(path.storage_options) == 0 else dict(path.storage_options)


def _get_default_variable_name(container: xr_Dataset | zarr.Group) -> str:
    """Return the default variable name from dataset or Zarr group attributes."""
    attributes = container.attrs.get("attributes")
    var_name = attributes.get("defaultVariableName") if isinstance(attributes, Mapping) else None
    if not isinstance(var_name, str) or var_name not in container:
        msg = "Default variable name is missing or not found in MDIO dataset."
        raise MDIOMissingVariableError(msg)
    return var_name


def _create_empty_mask_like(
    group: zarr.Group,
    name: str,
    template_array: zarr.Array,
    attributes: dict[str, object],
    *,
    overwrite: bool = False,
) -> zarr.Array:
    """Create a boolean Zarr array initialized to False with the same layout as ``template_array``."""
    is_v2 = group.metadata.zarr_format == ZarrFormat.V2
    dims = (
        list(template_array.attrs.get("_ARRAY_DIMENSIONS", ()))
        if is_v2
        else list(template_array.metadata.dimension_names or ())
    )

    array_attrs = dict(attributes)
    if "coordinates" in template_array.attrs:
        array_attrs["coordinates"] = template_array.attrs["coordinates"]

    if is_v2:
        array_attrs["_ARRAY_DIMENSIONS"] = dims
        create_kwargs: dict[str, object] = {
            "compressors": Blosc(cname="zstd"),
            "dimension_separator": "/",
        }
    else:
        create_kwargs = {
            "compressors": BloscCodec(cname="zstd"),
            "dimension_names": dims,
        }

    with zarr_warnings_suppress_unstable_structs_v3():
        return group.create_array(
            name=name,
            shape=template_array.shape,
            chunks=template_array.chunks,
            dtype=bool,
            fill_value=False,
            attributes=array_attrs,
            overwrite=overwrite,
            **create_kwargs,
        )


@contextmanager
def _open_for_metadata_update(path: UPath) -> Generator[zarr.Group, None, None]:
    """Open an existing store to update its attributes.

    Xarray cannot persist attribute updates on an existing store, so attribute writes go
    through Zarr. Updates read unconsolidated node metadata as the source of truth so stale
    consolidated metadata cannot erase newer attributes. For Zarr v2 stores, consolidated
    metadata is rebuilt from the opened group's format after the body returns or raises.

    Args:
        path: Universal path of the MDIO store.

    Yields:
        The root Zarr group, opened for update.
    """
    group = zarr.open_group(
        path.as_posix(),
        mode="r+",
        storage_options=_normalize_storage_options(path),
        use_consolidated=False,
    )
    try:
        yield group
    finally:
        if group.metadata.zarr_format == ZarrFormat.V2:
            zarr.consolidate_metadata(group.store)


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
    zarr_format = zarr.config.get("default_zarr_format")

    return xr_open_zarr(
        input_path.as_posix(),
        chunks=chunks,
        storage_options=storage_options,
        mask_and_scale=zarr_format == ZarrFormat.V3,  # off for v2, on for v3
        consolidated=zarr_format == ZarrFormat.V2,  # on for v2, off for v3
    )


def to_mdio(  # noqa: PLR0913
    dataset: Dataset,
    output_path: UPath | Path | str,
    mode: ZarrWriteModes | None = None,
    *,
    compute: bool = True,
    region: Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None = None,
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
    """
    output_path = _normalize_path(output_path)
    storage_options = _normalize_storage_options(output_path)
    zarr_format = zarr.config.get("default_zarr_format")

    with zarr_warnings_suppress_unstable_structs_v3():
        xr_to_zarr(
            dataset,
            store=output_path.as_posix(),  # xarray doesn't like URI when file:// is protocol
            mode=mode,
            compute=compute,
            consolidated=zarr_format == ZarrFormat.V2,  # on for v2, off for v3
            region=region,
            storage_options=storage_options,
            write_empty_chunks=False,
        )


def update_crs(mdio_path: UPath | Path | str, crs: str | None) -> None:
    """Set or remove the coordinate reference system on an existing MDIO file.

    Replaces only the root ``crs`` attribute, preserving all other root and array
    attributes. Passing ``None`` removes only ``crs``. Array payloads are not read or
    rewritten. For Zarr v2 stores, unconsolidated node metadata is the source of truth
    and consolidated metadata is rebuilt from the opened group's format after the update.

    Args:
        mdio_path: Local or remote path to the MDIO store.
        crs: Replacement CRS string, e.g. ``"EPSG:32610"``, or ``None`` to remove it.

    Raises:
        MDIONotFoundError: If ``mdio_path`` does not exist.
    """
    path = _normalize_path(mdio_path)
    if not path.exists():
        msg = f"MDIO file not found: {path}"
        raise MDIONotFoundError(msg)

    with _open_for_metadata_update(path) as group:
        if crs is None:
            group.attrs.pop("crs", None)
        else:
            group.attrs["crs"] = crs
