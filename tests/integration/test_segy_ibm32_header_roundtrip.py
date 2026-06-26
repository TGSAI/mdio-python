"""Round-trip tests for SEG-Y files that declare IBM float (``ibm32``) trace-header fields.

These guard the IBM-real header data path end-to-end through real file I/O:

* Ingestion (Defect A): mdio used to persist an ``ibm32`` header in its raw ``uint32`` slot,
  casting the decoded float down to an integer. That truncated decimals (``118.625`` ->
  ``118``) and wrapped the sign of negatives (``-50.25`` -> a huge unsigned int). The fix
  promotes ``ibm32`` header fields to ``float32`` so the decoded value is stored faithfully.
* Export (Defect B): SegyFactory IBM-encodes ``ibm32`` *header* fields from ``segy`` 0.6.0
  onward (the minimum mdio requires), so the full round-trip exercises that encode/decode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import fsspec
import numpy as np
from segy.factory import SegyFactory
from segy.schema import HeaderField
from segy.standards import get_segy_standard

from mdio import mdio_to_segy
from mdio.api.io import open_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.converters.segy import segy_to_mdio
from mdio.segy.file import SegyFileWrapper

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from segy.schema import SegySpec

# Dyadic rationals chosen so they are exactly representable in IBM hex float, letting us assert
# exact equality. They cover the original failure modes: decimal truncation and sign wrapping.
IBM_HEADER_VALUES = np.array([118.625, -50.25, 0.5, -1.5], dtype="float32")

GRID_INLINES = 3
GRID_CROSSLINES = 4
NUM_SAMPLES = 8
IBM_FIELD_NAME = "ibm_attr"


def _ibm32_segy_spec() -> SegySpec:
    """Build a rev1 SegySpec with an ``ibm32`` trace-header field alongside index/coord fields."""
    fields = [
        HeaderField(name="inline", byte=189, format="int32"),
        HeaderField(name="crossline", byte=193, format="int32"),
        HeaderField(name="coordinate_scalar", byte=71, format="int16"),
        HeaderField(name="cdp_x", byte=181, format="int32"),
        HeaderField(name="cdp_y", byte=185, format="int32"),
        HeaderField(name="samples_per_trace", byte=115, format="int16"),
        HeaderField(name="sample_interval", byte=117, format="int16"),
        # Unassigned rev1 bytes 233-240: safe spot for a custom IBM-float attribute.
        HeaderField(name=IBM_FIELD_NAME, byte=233, format="ibm32"),
    ]
    spec = get_segy_standard(1).customize(trace_header_fields=fields)
    spec.segy_standard = 1
    return spec


def _ibm_values_per_trace(num_traces: int) -> np.ndarray:
    """Tile the sample IBM values to cover every trace."""
    reps = int(np.ceil(num_traces / IBM_HEADER_VALUES.size))
    return np.tile(IBM_HEADER_VALUES, reps)[:num_traces].astype("float32")


def _write_ibm32_segy(path: Path, spec: SegySpec) -> np.ndarray:
    """Write a small 3D SEG-Y whose ``ibm32`` header holds real IBM-float values.

    ``segy`` >= 0.6.0 exposes ``ibm32`` header fields as ``float32`` in the factory template and
    IBM-encodes them on write, so real float values are assigned directly.

    Args:
        path: Destination path for the generated SEG-Y file.
        spec: SegySpec describing the trace header layout (must declare the ibm32 field).

    Returns:
        The IBM header value assigned to each trace, in trace order.
    """
    num_traces = GRID_INLINES * GRID_CROSSLINES
    factory = SegyFactory(spec=spec, samples_per_trace=NUM_SAMPLES)
    samples = factory.create_trace_sample_template(num_traces)
    headers = factory.create_trace_header_template(num_traces)

    inlines, crosslines = np.mgrid[0:GRID_INLINES, 0:GRID_CROSSLINES]
    headers["inline"] = (10 + inlines).ravel()
    headers["crossline"] = (100 + crosslines * 2).ravel()
    headers["coordinate_scalar"] = -100
    headers["cdp_x"] = (700_000 + inlines * 100).ravel()
    headers["cdp_y"] = (4_000_000 + crosslines * 100).ravel()
    headers["samples_per_trace"] = NUM_SAMPLES
    headers["sample_interval"] = 4000

    ibm_values = _ibm_values_per_trace(num_traces)
    headers[IBM_FIELD_NAME] = ibm_values

    samples[:] = (np.arange(num_traces) + 1)[:, None]

    with fsspec.open(path.as_posix(), mode="wb") as fp:
        fp.write(factory.create_textual_header())
        fp.write(factory.create_binary_header())
        fp.write(factory.create_traces(headers, samples))

    return ibm_values


def test_ingested_ibm32_header_preserves_value(tmp_path: Path) -> None:
    """SEG-Y -> MDIO must store ibm32 headers as float32 without truncating or wrapping (Defect A)."""
    spec = _ibm32_segy_spec()
    segy_path = tmp_path / "ibm32.sgy"
    mdio_path = tmp_path / "ibm32.mdio"

    expected = _write_ibm32_segy(segy_path, spec)

    segy_to_mdio(
        segy_spec=spec,
        mdio_template=TemplateRegistry().get("PostStack3DTime"),
        input_path=segy_path,
        output_path=mdio_path,
        overwrite=True,
    )

    headers = open_mdio(mdio_path)["headers"].values
    stored = headers[IBM_FIELD_NAME].ravel()

    assert stored.dtype == np.float32
    np.testing.assert_array_equal(stored, expected)


def test_ibm32_header_full_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEG-Y -> MDIO -> SEG-Y must preserve ibm32 header values end-to-end."""
    # The SEG-Y file headers must be persisted at ingest so they can be re-emitted on export.
    monkeypatch.setenv("MDIO__IMPORT__SAVE_SEGY_FILE_HEADER", "true")

    spec = _ibm32_segy_spec()
    segy_path = tmp_path / "ibm32.sgy"
    mdio_path = tmp_path / "ibm32.mdio"
    segy_rt_path = tmp_path / "ibm32_rt.sgy"

    expected = _write_ibm32_segy(segy_path, spec)

    segy_to_mdio(
        segy_spec=spec,
        mdio_template=TemplateRegistry().get("PostStack3DTime"),
        input_path=segy_path,
        output_path=mdio_path,
        overwrite=True,
    )
    mdio_to_segy(segy_spec=spec, input_path=mdio_path, output_path=segy_rt_path)

    roundtripped = SegyFileWrapper(segy_rt_path.as_posix(), spec=spec).trace[:].header[IBM_FIELD_NAME]
    np.testing.assert_array_equal(roundtripped, expected)
