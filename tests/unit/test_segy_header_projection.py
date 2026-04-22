"""Tests for SEG-Y trace header projection used during MDIO to SEG-Y export."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import SegySpec
from segy.schema import TraceDataSpec
from segy.schema import TraceSpec
from segy.standards import get_segy_standard

from mdio.segy.utilities import project_headers_to_segy_spec


def _make_segy_spec(fields: list[HeaderField]) -> SegySpec:
    """Build a SegySpec whose trace header contains exactly the given fields."""
    base = get_segy_standard(1.0)
    trace = TraceSpec(
        header=HeaderSpec(fields=fields, item_size=240),
        data=TraceDataSpec(format="ibm32"),
    )
    return SegySpec(
        segy_standard=base.segy_standard,
        text_header=base.text_header,
        binary_header=base.binary_header,
        trace=trace,
    )


def _mdio_headers(num: int) -> np.ndarray:
    """Create a structured array simulating MDIO-stored headers (superset of fields)."""
    dtype = np.dtype(
        [
            ("inline", "<i4"),
            ("crossline", "<i4"),
            ("cdp_x", "<i4"),
            ("cdp_y", "<i4"),
            ("samples_per_trace", "<i2"),
            ("sample_interval", "<i2"),
        ]
    )
    headers = np.zeros(num, dtype=dtype)
    headers["inline"] = np.arange(num, dtype=np.int32)
    headers["crossline"] = np.arange(num, dtype=np.int32) * 2
    headers["cdp_x"] = np.arange(num, dtype=np.int32) * 10
    headers["cdp_y"] = np.arange(num, dtype=np.int32) * 20
    headers["samples_per_trace"] = 201
    headers["sample_interval"] = 4000
    return headers


class TestProjectHeadersToSegySpec:
    """Cases covering subset, reorder, missing and preservation semantics."""

    def test_subset_spec_returns_only_requested_fields(self) -> None:
        """A SegySpec with fewer fields than MDIO yields only those fields."""
        headers = da.from_array(_mdio_headers(8), chunks=4)
        spec = _make_segy_spec(
            [
                HeaderField(name="inline", byte=189, format="int32"),
                HeaderField(name="crossline", byte=193, format="int32"),
            ]
        )

        projected = project_headers_to_segy_spec(headers, spec).compute()

        assert list(projected.dtype.names) == ["inline", "crossline"]
        np.testing.assert_array_equal(projected["inline"], np.arange(8, dtype=np.int32))
        np.testing.assert_array_equal(projected["crossline"], np.arange(8, dtype=np.int32) * 2)

    def test_reordered_spec_preserves_values_per_field_name(self) -> None:
        """Reordering SegySpec fields keeps the per-name values intact."""
        headers = da.from_array(_mdio_headers(6), chunks=3)
        # Deliberately reverse the MDIO storage order.
        spec = _make_segy_spec(
            [
                HeaderField(name="cdp_y", byte=185, format="int32"),
                HeaderField(name="cdp_x", byte=181, format="int32"),
                HeaderField(name="crossline", byte=193, format="int32"),
                HeaderField(name="inline", byte=189, format="int32"),
            ]
        )

        projected = project_headers_to_segy_spec(headers, spec).compute()

        assert list(projected.dtype.names) == ["cdp_y", "cdp_x", "crossline", "inline"]
        np.testing.assert_array_equal(projected["inline"], np.arange(6, dtype=np.int32))
        np.testing.assert_array_equal(projected["crossline"], np.arange(6, dtype=np.int32) * 2)
        np.testing.assert_array_equal(projected["cdp_x"], np.arange(6, dtype=np.int32) * 10)
        np.testing.assert_array_equal(projected["cdp_y"], np.arange(6, dtype=np.int32) * 20)

    def test_missing_field_raises_value_error(self) -> None:
        """Requesting a field not present in MDIO raises ValueError."""
        headers = da.from_array(_mdio_headers(4), chunks=2)
        spec = _make_segy_spec(
            [
                HeaderField(name="inline", byte=189, format="int32"),
                HeaderField(name="offset", byte=37, format="int32"),
            ]
        )

        with pytest.raises(ValueError, match="offset"):
            project_headers_to_segy_spec(headers, spec)

    def test_output_dtype_is_packed_native_matching_spec_field_order(self) -> None:
        """Projected dtype is packed (no gaps), native-order, and ordered like the SegySpec.

        A packed dtype sidesteps numpy byteswap artifacts on structured arrays with
        explicit field offsets / padding bytes. Downstream SegyFactory assignment relies on
        positional (not name-based) copying, so ordering must match the SegySpec.
        """
        headers = da.from_array(_mdio_headers(2), chunks=2)
        spec = _make_segy_spec(
            [
                HeaderField(name="inline", byte=189, format="int32"),
                HeaderField(name="crossline", byte=193, format="int32"),
            ]
        )

        projected = project_headers_to_segy_spec(headers, spec)

        spec_names = list(spec.trace.header.dtype.names)
        assert list(projected.dtype.names) == spec_names
        packed_itemsize = sum(
            spec.trace.header.dtype.fields[name][0].itemsize for name in spec_names
        )
        assert projected.dtype.itemsize == packed_itemsize
        for name in spec_names:
            spec_field_dtype = spec.trace.header.dtype.fields[name][0]
            assert projected.dtype.fields[name][0] == spec_field_dtype.newbyteorder("=")

    def test_projection_independent_of_mdio_field_order(self) -> None:
        """Reordering the MDIO source fields must not change projected values."""
        original = _mdio_headers(5)
        shuffled_dtype = np.dtype(
            [
                ("cdp_y", "<i4"),
                ("sample_interval", "<i2"),
                ("inline", "<i4"),
                ("crossline", "<i4"),
                ("cdp_x", "<i4"),
                ("samples_per_trace", "<i2"),
            ]
        )
        shuffled = np.empty(original.shape, dtype=shuffled_dtype)
        for name in shuffled_dtype.names:
            shuffled[name] = original[name]

        spec = _make_segy_spec(
            [
                HeaderField(name="inline", byte=189, format="int32"),
                HeaderField(name="crossline", byte=193, format="int32"),
                HeaderField(name="cdp_x", byte=181, format="int32"),
                HeaderField(name="cdp_y", byte=185, format="int32"),
            ]
        )

        a = project_headers_to_segy_spec(da.from_array(original, chunks=5), spec).compute()
        b = project_headers_to_segy_spec(da.from_array(shuffled, chunks=5), spec).compute()

        for name in spec.trace.header.names:
            np.testing.assert_array_equal(a[name], b[name])

    def test_short_circuit_on_matching_dtype(self) -> None:
        """If source headers dtype exactly matches target dtype, return array unchanged."""
        spec = _make_segy_spec(
            [
                HeaderField(name="inline", byte=189, format="int32"),
                HeaderField(name="crossline", byte=193, format="int32"),
            ]
        )

        target_dtype = np.dtype(
            [
                ("inline", spec.trace.header.dtype.fields["inline"][0].newbyteorder("=")),
                ("crossline", spec.trace.header.dtype.fields["crossline"][0].newbyteorder("=")),
            ]
        )

        # Create a dask array that perfectly matches the target_dtype
        headers_np = np.zeros(5, dtype=target_dtype)
        headers_np["inline"] = np.arange(5)
        headers_np["crossline"] = np.arange(5) * 2
        headers_da = da.from_array(headers_np, chunks=5)

        projected_da = project_headers_to_segy_spec(headers_da, spec)

        # The exact same dask array object should be returned (no map_blocks overhead)
        assert projected_da is headers_da

