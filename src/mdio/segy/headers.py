"""In-place updates for SEG-Y text and binary file headers on an MDIO dataset.

Headers live as attributes on the scalar ``segy_file_header`` variable. Reads use
:func:`mdio.open_mdio`. Writes go through Zarr because xarray does not persist attribute
updates on an existing array.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import zarr
from segy.factory import SegyFactory
from segy.factory import get_default_text
from segy.standards import get_segy_standard

from mdio.api.io import _normalize_path
from mdio.api.io import _normalize_storage_options
from mdio.api.io import open_mdio
from mdio.constants import ZarrFormat
from mdio.core.zarr_io import zarr_warnings_suppress_unstable_structs_v3
from mdio.exceptions import MDIONotFoundError
from mdio.segy.text_header import sanitize_text_header
from mdio.segy.text_header import validate_text_header

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from segy.schema import SegySpec
    from upath import UPath
    from xarray import Dataset as xr_Dataset
    from zarr import Array as ZarrArray
    from zarr import Group as ZarrGroup


logger = logging.getLogger(__name__)

SEGY_FILE_HEADER_VARIABLE = "segy_file_header"
TEXT_HEADER_ATTR = "textHeader"
BINARY_HEADER_ATTR = "binaryHeader"

_DEFAULT_SAMPLE_INTERVAL = 4000
_DEFAULT_SAMPLES_PER_TRACE = 1500


@dataclass(frozen=True, slots=True)
class SegyFileHeaders:
    """Resolved SEG-Y text and binary file headers stored on an MDIO dataset.

    Attributes:
        text_header: Sanitized 40x80 textual file header.
        binary_header: Binary header fields as a JSON-friendly integer mapping.
    """

    text_header: str
    binary_header: dict[str, int]


def update_segy_file_headers(
    mdio_path: UPath | Path | str,
    *,
    text_header: str | None = None,
    binary_header: Mapping[str, int] | None = None,
    template: SegySpec | None = None,
) -> SegyFileHeaders:
    """Update SEG-Y text and/or binary headers on an existing MDIO file.

    Reads the dataset with :func:`mdio.open_mdio` (lazy chunks) and writes only
    ``segy_file_header`` attributes. Trace data is not read or rewritten.

    ``None`` arguments leave that header unchanged when it already exists. If the variable
    or an attribute is missing, a default is written. Defaults come from ``template`` when
    given, otherwise SEG-Y Revision 1.0. ``samples_per_trace`` is taken from the default
    data variable shape when that metadata is present.

    User ``binary_header`` values are merged onto the existing header, or onto the default
    header when the file has none. Prefer updating values of keys that already exist.
    Adding or removing binary fields can break SEG-Y export.

    Args:
        mdio_path: Local or remote path to the MDIO store.
        text_header: Replacement textual header. Sanitized to the 40x80 ASCII card layout.
        binary_header: Binary header fields to set or overlay. Integer-valued mapping.
        template: SEG-Y spec used to build defaults when the file lacks header fields.

    Returns:
        The text and binary headers stored after the update.

    Raises:
        MDIONotFoundError: If ``mdio_path`` does not exist.
        ValueError: If a provided text header cannot be sanitized or a binary field is not
            an integer.
    """
    path = _normalize_path(mdio_path)
    if not path.exists():
        msg = f"MDIO file not found: {path}"
        raise MDIONotFoundError(msg)
    dataset = open_mdio(path, chunks={})

    existing_text = _read_text_header(dataset)
    existing_binary = _read_binary_header(dataset)
    user_text = sanitize_text_header(text_header) if text_header is not None else None
    user_binary = _coerce_binary_header(binary_header) if binary_header is not None else None
    if user_binary is not None:
        _warn_unknown_binary_keys(user_binary, template)

    default_text: str | None = None
    default_binary: dict[str, int] | None = None
    if (user_text is None and existing_text is None) or existing_binary is None:
        spec = _resolve_spec(template)
        sample_interval, samples_per_trace = _resolve_factory_params(dataset, existing_binary)
        default_text, default_binary = _build_defaults(spec, sample_interval, samples_per_trace)

    if user_text is not None:
        resolved_text = user_text
    elif existing_text is not None:
        resolved_text = existing_text
    else:
        resolved_text = default_text

    base_binary = existing_binary if existing_binary is not None else default_binary
    if resolved_text is None or base_binary is None:
        msg = "Failed to resolve SEG-Y file headers"
        raise ValueError(msg)
    resolved_binary = {**base_binary, **user_binary} if user_binary is not None else base_binary

    validate_text_header(resolved_text)
    _write_header_attrs(path, resolved_text, resolved_binary)
    return SegyFileHeaders(text_header=resolved_text, binary_header=resolved_binary)


def _read_text_header(dataset: xr_Dataset) -> str | None:
    """Return the stored text header when it is a string."""
    if SEGY_FILE_HEADER_VARIABLE not in dataset:
        return None
    value = dataset[SEGY_FILE_HEADER_VARIABLE].attrs.get(TEXT_HEADER_ATTR)
    return value if isinstance(value, str) else None


def _read_binary_header(dataset: xr_Dataset) -> dict[str, int] | None:
    """Return the stored binary header when it is a mapping of integers."""
    if SEGY_FILE_HEADER_VARIABLE not in dataset:
        return None
    value = dataset[SEGY_FILE_HEADER_VARIABLE].attrs.get(BINARY_HEADER_ATTR)
    if not isinstance(value, dict):
        return None
    return _coerce_binary_header(value)


def _resolve_spec(template: SegySpec | None) -> SegySpec:
    """Return a writable spec copy. Rev 1.0 when the caller did not pass a template."""
    if template is None:
        return get_segy_standard(1.0)
    return template.model_copy(deep=True)


def _resolve_factory_params(dataset: xr_Dataset, existing_binary: dict[str, int] | None) -> tuple[int, int]:
    """Pick sample interval and samples-per-trace for default header generation."""
    sample_interval = _DEFAULT_SAMPLE_INTERVAL
    samples_per_trace = _infer_samples_per_trace(dataset) or _DEFAULT_SAMPLES_PER_TRACE
    if existing_binary is not None:
        sample_interval = existing_binary.get("sample_interval", sample_interval)
        samples_per_trace = existing_binary.get("samples_per_trace", samples_per_trace)
    return sample_interval, samples_per_trace


def _infer_samples_per_trace(dataset: xr_Dataset) -> int | None:
    """Return the last-axis length of the default data variable, if present."""
    attributes = dataset.attrs.get("attributes")
    if not isinstance(attributes, dict):
        return None
    variable_name = attributes.get("defaultVariableName")
    if not isinstance(variable_name, str) or variable_name not in dataset:
        return None
    data = dataset[variable_name]
    if not data.dims:
        return None
    return int(data.sizes[data.dims[-1]])


def _build_defaults(spec: SegySpec, sample_interval: int, samples_per_trace: int) -> tuple[str, dict[str, int]]:
    """Build default text and binary headers from a SEG-Y spec."""
    factory = SegyFactory(spec=spec, sample_interval=sample_interval, samples_per_trace=samples_per_trace)
    text_header = get_default_text(factory.spec)
    binary_header = _binary_bytes_to_mdio_dict(factory.create_binary_header(), factory.spec)
    return text_header, binary_header


def _binary_bytes_to_mdio_dict(raw: bytes, spec: SegySpec) -> dict[str, int]:
    """Decode factory binary-header bytes into the MDIO attribute mapping."""
    parsed = np.frombuffer(raw, dtype=spec.binary_header.dtype)
    header = {name: int(parsed[name][0]) for name in parsed.dtype.names or ()}
    return _normalize_revision_keys(header)


def _normalize_revision_keys(binary_header: dict[str, int]) -> dict[str, int]:
    """Store revision as major/minor, matching SEG-Y ingest."""
    normalized = dict(binary_header)
    if "segy_revision_major" in normalized and "segy_revision_minor" in normalized:
        normalized.pop("segy_revision", None)
        return normalized
    if "segy_revision" in normalized:
        code = normalized.pop("segy_revision")
        normalized["segy_revision_major"] = (code >> 8) & 0xFF
        normalized["segy_revision_minor"] = code & 0xFF
    return normalized


def _coerce_binary_header(binary_header: Mapping[str, object]) -> dict[str, int]:
    """Integer-coerce a user or stored binary header mapping."""
    coerced: dict[str, int] = {}
    for key, value in binary_header.items():
        try:
            coerced[key] = int(value)
        except (TypeError, ValueError) as exc:
            msg = f"Binary header field {key!r} must be an integer, got {value!r}"
            raise ValueError(msg) from exc
    return _normalize_revision_keys(coerced)


def _warn_unknown_binary_keys(user_binary: dict[str, int], template: SegySpec | None) -> None:
    """Warn when the user adds fields the SEG-Y template cannot encode."""
    spec = _resolve_spec(template)
    known = set(spec.binary_header.names)
    known.discard("segy_revision")
    known.update({"segy_revision_major", "segy_revision_minor"})
    unknown = sorted(set(user_binary) - known)
    if unknown:
        logger.warning(
            "Binary header fields %s are not in the SEG-Y template; adding or removing fields can break export.",
            unknown,
        )


def _write_header_attrs(path: UPath, text_header: str, binary_header: dict[str, int]) -> None:
    """Persist header attributes through Zarr."""
    storage_options = _normalize_storage_options(path)
    zarr_format = zarr.config.get("default_zarr_format")
    group = zarr.open_group(
        path.as_posix(),
        mode="r+",
        storage_options=storage_options,
        use_consolidated=zarr_format == ZarrFormat.V2,
    )
    header_array = _ensure_header_array(group)
    header_array.attrs.update({TEXT_HEADER_ATTR: text_header, BINARY_HEADER_ATTR: binary_header})
    if zarr_format == ZarrFormat.V2:
        zarr.consolidate_metadata(group.store)


def _ensure_header_array(group: ZarrGroup) -> ZarrArray:
    """Return the scalar header variable, creating it when the store has none."""
    if SEGY_FILE_HEADER_VARIABLE in group:
        return group[SEGY_FILE_HEADER_VARIABLE]
    with zarr_warnings_suppress_unstable_structs_v3():
        return group.create_array(SEGY_FILE_HEADER_VARIABLE, shape=(), dtype="U1", fill_value="")
