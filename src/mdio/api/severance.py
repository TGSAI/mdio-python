"""Z-severance mask creation, locking, resolution, and application for MDIO datasets."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
from pydantic import AwareDatetime
from pydantic import ValidationError
from pydantic import field_validator
from xarray import where as xr_where

from mdio.api.io import _create_empty_mask_like
from mdio.api.io import _get_default_variable_name
from mdio.api.io import _normalize_path
from mdio.api.io import _open_for_metadata_update
from mdio.builder.schemas.core import CamelCaseStrictModel
from mdio.exceptions import MDIOAlreadyExistsError
from mdio.exceptions import MDIOMissingVariableError
from mdio.exceptions import MDIONotFoundError
from mdio.exceptions import MDIOSeveranceError

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping
    from pathlib import Path

    from upath import UPath
    from xarray import DataArray as xr_DataArray
    from xarray import Dataset as xr_Dataset
    from zarr import Array as ZarrArray


DEFAULT_SEVERANCE_MASK_NAME = "z_severance_mask"
MASK_TYPE_ATTR = "maskType"
Z_SEVERANCE_MASK_TYPE: Literal["z_severance"] = "z_severance"
CREATED_ON_ATTR = "createdOn"
METADATA_LOCK_ATTR = "metadataLock"

_SEVERANCE_METADATA_KEYS = (MASK_TYPE_ATTR, CREATED_ON_ATTR, METADATA_LOCK_ATTR)


class SeveranceMaskMetadata(CamelCaseStrictModel):
    """Canonical metadata stored on a z-severance mask variable."""

    mask_type: Literal["z_severance"]
    created_on: AwareDatetime
    metadata_lock: bool

    @field_validator("created_on", mode="after")
    @classmethod
    def _normalize_utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)


def _build_mask_metadata(created_on: datetime | str | None = None) -> SeveranceMaskMetadata:
    """Validate and construct canonical locked severance mask metadata."""
    try:
        return SeveranceMaskMetadata(
            mask_type=Z_SEVERANCE_MASK_TYPE,
            created_on=datetime.now(UTC) if created_on is None else created_on,
            metadata_lock=True,
        )
    except ValidationError as exc:
        msg = f"Invalid z-severance mask metadata: {exc}"
        raise MDIOSeveranceError(msg) from exc


def _parse_mask_metadata(attrs: Mapping[str, object], mask_name: str) -> SeveranceMaskMetadata:
    """Parse and validate canonical severance mask metadata from variable attributes."""
    if attrs.get(MASK_TYPE_ATTR) != Z_SEVERANCE_MASK_TYPE:
        msg = f"Variable {mask_name!r} is not a z-severance mask (expected {MASK_TYPE_ATTR}={Z_SEVERANCE_MASK_TYPE!r})."
        raise MDIOSeveranceError(msg)

    raw_subset = {key: attrs[key] for key in _SEVERANCE_METADATA_KEYS if key in attrs}
    try:
        return SeveranceMaskMetadata.model_validate(raw_subset)
    except ValidationError as exc:
        msg = f"Severance mask {mask_name!r} has invalid metadata: {exc}"
        raise MDIOSeveranceError(msg) from exc


def _validate_severance_mask_array(
    mask_name: str,
    da: xr_DataArray,
    default_var: xr_DataArray,
) -> datetime:
    """Validate a severance mask DataArray and return its parsed UTC creation timestamp."""
    meta = _parse_mask_metadata(da.attrs, mask_name)
    if meta.metadata_lock:
        msg = f"Severance mask {mask_name!r} has an active metadata lock; mask population did not complete."
        raise MDIOSeveranceError(msg)

    if da.dtype != np.bool_:
        msg = f"Severance mask {mask_name!r} must have boolean dtype, got {da.dtype}."
        raise MDIOSeveranceError(msg)

    if da.dims != default_var.dims or da.shape != default_var.shape:
        msg = (
            f"Severance mask {mask_name!r} dimensions/shape ({da.dims}, {da.shape}) "
            f"do not match default variable ({default_var.dims}, {default_var.shape})."
        )
        raise MDIOSeveranceError(msg)

    return meta.created_on


def create_z_severance_mask(
    mdio_path: UPath | Path | str,
    mask_name: str = DEFAULT_SEVERANCE_MASK_NAME,
    *,
    created_on: datetime | str | None = None,
    overwrite: bool = False,
) -> ZarrArray:
    """Create a new N-D boolean z-severance mask initialized to False in a locked state.

    The mask variable is created with the exact dimensions, shape, chunk grid, and non-dimension
    coordinates of the dataset's default variable. Chunks are lazily initialized to ``False``
    via Zarr ``fill_value=False`` so no dense chunk files are written until severed sections are
    populated. ``metadataLock: true`` is written to the mask's JSON metadata until
    :func:`finalize_z_severance_mask` is called.

    Args:
        mdio_path: Local or remote path to the MDIO store.
        mask_name: Name of the severance mask variable to create.
        created_on: Optional timezone-aware creation timestamp (defaults to current UTC time).
        overwrite: Whether to overwrite an existing severance mask variable with the same name.

    Returns:
        Opened writable Zarr Array for the newly created severance mask.

    Raises:
        MDIONotFoundError: If ``mdio_path`` does not exist.
        MDIOAlreadyExistsError: If ``mask_name`` already exists and ``overwrite=False``.
        MDIOSeveranceError: If ``mask_name`` collides with a non-mask variable or ``created_on``
            is invalid.
    """
    path = _normalize_path(mdio_path)
    if not path.exists():
        msg = f"MDIO file not found: {path}"
        raise MDIONotFoundError(msg)

    if not mask_name or not mask_name.strip():
        msg = "'mask_name' must be a non-empty string."
        raise MDIOSeveranceError(msg)

    metadata = _build_mask_metadata(created_on)

    with _open_for_metadata_update(path) as group:
        default_var_name = _get_default_variable_name(group)

        if mask_name in group:
            if group[mask_name].attrs.get(MASK_TYPE_ATTR) != Z_SEVERANCE_MASK_TYPE:
                msg = f"Mask name {mask_name!r} collides with an existing non-mask variable or coordinate."
                raise MDIOSeveranceError(msg)
            if not overwrite:
                msg = f"Severance mask {mask_name!r} already exists in {path}."
                raise MDIOAlreadyExistsError(msg)

        return _create_empty_mask_like(
            group=group,
            name=mask_name,
            template_array=group[default_var_name],
            attributes=metadata.model_dump(mode="json"),
            overwrite=overwrite,
        )


def finalize_z_severance_mask(
    mdio_path: UPath | Path | str,
    mask_name: str = DEFAULT_SEVERANCE_MASK_NAME,
) -> None:
    """Release the metadata lock for a populated z-severance mask.

    Args:
        mdio_path: Local or remote path to the MDIO store.
        mask_name: Name of the severance mask variable to finalize.

    Raises:
        MDIONotFoundError: If ``mdio_path`` does not exist.
        MDIOMissingVariableError: If ``mask_name`` does not exist in the store.
    """
    path = _normalize_path(mdio_path)
    if not path.exists():
        msg = f"MDIO file not found: {path}"
        raise MDIONotFoundError(msg)

    with _open_for_metadata_update(path) as group:
        if mask_name not in group:
            msg = f"Severance mask variable {mask_name!r} not found in MDIO dataset."
            raise MDIOMissingVariableError(msg)

        mask_array = group[mask_name]
        _parse_mask_metadata(mask_array.attrs, mask_name)
        mask_array.attrs[METADATA_LOCK_ATTR] = False


@contextmanager
def z_severance_mask_session(
    mdio_path: UPath | Path | str,
    mask_name: str = DEFAULT_SEVERANCE_MASK_NAME,
    *,
    created_on: datetime | str | None = None,
    overwrite: bool = False,
) -> Generator[ZarrArray, None, None]:
    """Context manager that creates a locked z-severance mask and unlocks it only on clean exit.

    If an exception is raised inside the ``with`` block, ``metadataLock`` remains ``True`` in JSON
    so downstream exports fail closed rather than exporting an incomplete cut.

    Args:
        mdio_path: Local or remote path to the MDIO store.
        mask_name: Name of the severance mask variable to create.
        created_on: Optional timezone-aware creation timestamp.
        overwrite: Whether to overwrite an existing mask variable with the same name.

    Yields:
        Opened writable Zarr Array for populating severed regions with ``True``.
    """
    mask_array = create_z_severance_mask(
        mdio_path=mdio_path,
        mask_name=mask_name,
        created_on=created_on,
        overwrite=overwrite,
    )
    yield mask_array
    finalize_z_severance_mask(mdio_path=mdio_path, mask_name=mask_name)


def resolve_z_severance_mask(
    dataset: xr_Dataset,
    severance_mask: str | None = None,
) -> str | None:
    """Resolve and validate the z-severance mask variable name to apply for a dataset.

    Args:
        dataset: Opened MDIO Xarray Dataset.
        severance_mask: Explicit severance mask variable name, or ``None`` to select the latest
            mask by ``createdOn`` timestamp if any severance masks exist on the dataset.

    Returns:
        The resolved severance mask variable name, or ``None`` if no severance masks exist and
        ``severance_mask`` was ``None``.

    Raises:
        MDIOMissingVariableError: If an explicit ``severance_mask`` is not present in ``dataset``.
        MDIOSeveranceError: If the selected mask (or any candidate mask during automatic selection)
            is locked, has invalid dtype/dimensions/shape/metadata, or if ``createdOn`` timestamps
            tie.
    """
    default_var_name = _get_default_variable_name(dataset)
    default_var = dataset[default_var_name]

    if severance_mask is not None:
        if not severance_mask.strip():
            msg = "'severance_mask' must be a non-empty string when provided."
            raise MDIOSeveranceError(msg)
        if severance_mask not in dataset.data_vars:
            msg = f"Severance mask variable {severance_mask!r} not found in MDIO dataset."
            raise MDIOMissingVariableError(msg)
        _validate_severance_mask_array(severance_mask, dataset[severance_mask], default_var)
        return severance_mask

    validated = [
        (_validate_severance_mask_array(name, da, default_var), name)
        for name, da in dataset.data_vars.items()
        if da.attrs.get(MASK_TYPE_ATTR) == Z_SEVERANCE_MASK_TYPE
    ]
    if not validated:
        return None

    validated.sort(key=lambda item: item[0], reverse=True)
    if len(validated) > 1 and validated[0][0] == validated[1][0]:
        msg = (
            f"Multiple severance masks ({validated[0][1]!r}, {validated[1][1]!r}) share the same "
            f"latest {CREATED_ON_ATTR!r} timestamp ({validated[0][0].isoformat()}). "
            "Specify 'severance_mask' explicitly."
        )
        raise MDIOSeveranceError(msg)

    return validated[0][1]


def apply_z_severance_mask(
    dataset: xr_Dataset,
    mask_name: str,
) -> xr_Dataset:
    """Zero out the default variable where the specified z-severance mask is True.

    Args:
        dataset: Opened MDIO Xarray Dataset.
        mask_name: Resolved z-severance mask variable name present in ``dataset``.

    Returns:
        Dataset with severed samples set to zero on the default variable.
    """
    default_var_name = _get_default_variable_name(dataset)
    masked_ds = dataset.copy()
    target_da = masked_ds[default_var_name]
    mask_da = masked_ds[mask_name]

    aligned_mask = mask_da
    if target_da.chunks is not None:
        dim_chunks = dict(zip(target_da.dims, target_da.chunks, strict=True))
        aligned_mask = mask_da.chunk(dim_chunks)

    zero = np.zeros((), dtype=target_da.dtype)
    masked_ds[default_var_name] = xr_where(aligned_mask, zero, target_da, keep_attrs=True)
    masked_ds[default_var_name].encoding = dict(target_da.encoding)

    return masked_ds
