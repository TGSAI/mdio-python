"""Unit tests for z-severance mask creation, locking, resolution, and export."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
import zarr
from segy.factory import SegyFactory
from segy.schema import HeaderField
from segy.standards import get_segy_standard

from mdio import apply_z_severance_mask
from mdio import create_z_severance_mask
from mdio import finalize_z_severance_mask
from mdio import mdio_to_segy
from mdio import open_mdio
from mdio import resolve_z_severance_mask
from mdio import segy_to_mdio
from mdio import to_mdio
from mdio import z_severance_mask_session
from mdio.api.severance import CREATED_ON_ATTR
from mdio.api.severance import MASK_TYPE_ATTR
from mdio.api.severance import METADATA_LOCK_ATTR
from mdio.api.severance import Z_SEVERANCE_MASK_TYPE
from mdio.api.severance import SeveranceMaskMetadata
from mdio.builder.template_registry import TemplateRegistry
from mdio.exceptions import MDIOAlreadyExistsError
from mdio.exceptions import MDIOMissingVariableError
from mdio.exceptions import MDIONotFoundError
from mdio.exceptions import MDIOSeveranceError
from mdio.segy.file import SegyFileWrapper

if TYPE_CHECKING:
    from pathlib import Path
    from segy.schema import SegySpec


def _write_test_store(path: Path) -> Path:
    """Create a 3D MDIO store with inline, crossline, time, and cdp_x/cdp_y coordinates."""
    amplitudes = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4)) + 1.0
    cdp_x = np.array([[100.0, 200.0, 300.0], [100.0, 200.0, 300.0]], dtype=np.float64)
    cdp_y = np.array([[1000.0, 1000.0, 1000.0], [2000.0, 2000.0, 2000.0]], dtype=np.float64)
    trace_mask = np.ones((2, 3), dtype=bool)

    dataset = xr.Dataset(
        {
            "amplitude": (("inline", "crossline", "time"), amplitudes),
            "trace_mask": (("inline", "crossline"), trace_mask),
        },
        coords={
            "inline": ("inline", np.array([10, 20], dtype=np.int32)),
            "crossline": ("crossline", np.array([100, 200, 300], dtype=np.int32)),
            "time": ("time", np.array([0, 4, 8, 12], dtype=np.int32)),
            "cdp_x": (("inline", "crossline"), cdp_x),
            "cdp_y": (("inline", "crossline"), cdp_y),
        },
        attrs={"attributes": {"defaultVariableName": "amplitude"}, "name": "test_survey"},
    )
    to_mdio(dataset, path, mode="w")
    return path


def _write_tiny_3d_segy(path: Path) -> SegySpec:
    """Write a minimal 2x2x8 3D post-stack SEG-Y file for export testing."""
    spec = get_segy_standard(1.0).customize(
        trace_header_fields=[
            HeaderField(name="inline", byte=189, format="int32"),
            HeaderField(name="crossline", byte=193, format="int32"),
            HeaderField(name="cdp_x", byte=181, format="int32"),
            HeaderField(name="cdp_y", byte=185, format="int32"),
            HeaderField(name="coordinate_scalar", byte=71, format="int16"),
            HeaderField(name="samples_per_trace", byte=115, format="int16"),
            HeaderField(name="sample_interval", byte=117, format="int16"),
        ]
    )
    spec.segy_standard = 1.0
    factory = SegyFactory(spec=spec, sample_interval=4000, samples_per_trace=8)

    headers = factory.create_trace_header_template(4)
    headers["inline"] = [10, 10, 20, 20]
    headers["crossline"] = [100, 200, 100, 200]
    headers["cdp_x"] = [1000, 2000, 1000, 2000]
    headers["cdp_y"] = [5000, 5000, 6000, 6000]
    headers["coordinate_scalar"] = 1

    samples = factory.create_trace_sample_template(4)
    for idx in range(4):
        samples[idx, :] = float(idx + 1)

    with path.open("wb") as fp:
        fp.write(factory.create_textual_header())
        fp.write(factory.create_binary_header())
        fp.write(factory.create_traces(headers, samples))

    return spec


class TestSeveranceMaskMetadataModel:
    """Tests for ``SeveranceMaskMetadata`` serialization and validation."""

    def test_serializes_to_canonical_camel_case_json(self) -> None:
        """``SeveranceMaskMetadata`` normalizes timestamps to UTC and dumps camelCase keys."""
        ts = datetime(2026, 11, 9, 14, 0, 0, tzinfo=timezone(timedelta(hours=2)))
        meta = SeveranceMaskMetadata(mask_type="z_severance", created_on=ts, metadata_lock=True)
        dumped = meta.model_dump(mode="json")
        assert dumped == {
            "maskType": "z_severance",
            "createdOn": "2026-11-09T12:00:00Z",
            "metadataLock": True,
        }


class TestCreateAndFinalizeZSeveranceMask:
    """Tests for creating, locking, and finalizing z-severance masks."""

    def test_missing_store_raises(self, tmp_path: Path) -> None:
        """Creating or finalizing on a missing path raises ``MDIONotFoundError``."""
        missing = tmp_path / "missing.mdio"
        with pytest.raises(MDIONotFoundError):
            create_z_severance_mask(missing)
        with pytest.raises(MDIONotFoundError):
            finalize_z_severance_mask(missing)

    def test_creates_mask_initialized_to_false_with_lock_and_created_on(self, tmp_path: Path) -> None:
        """New mask is N-D bool initialized to False with ``createdOn`` and ``metadataLock``."""
        store = _write_test_store(tmp_path / "survey.mdio")
        ts = datetime(2026, 11, 9, 15, 30, 0, tzinfo=UTC)

        zarr_mask = create_z_severance_mask(store, "z_severance_v1", created_on=ts)
        assert zarr_mask.shape == (2, 3, 4)
        assert zarr_mask.dtype == np.bool_

        ds = open_mdio(store)
        assert "z_severance_v1" in ds.data_vars
        mask_da = ds["z_severance_v1"]
        assert mask_da.dims == ("inline", "crossline", "time")
        assert not mask_da.values.any()
        assert mask_da.attrs[CREATED_ON_ATTR] == "2026-11-09T15:30:00Z"
        assert mask_da.attrs[MASK_TYPE_ATTR] == Z_SEVERANCE_MASK_TYPE
        assert mask_da.attrs[METADATA_LOCK_ATTR] is True

        # Resolving while locked must fail closed
        with pytest.raises(MDIOSeveranceError, match="metadata lock"):
            resolve_z_severance_mask(ds)

        # Finalize unlocks the mask
        finalize_z_severance_mask(store, "z_severance_v1")
        ds_after = open_mdio(store)
        assert ds_after["z_severance_v1"].attrs[METADATA_LOCK_ATTR] is False
        assert resolve_z_severance_mask(ds_after) == "z_severance_v1"

    def test_session_unlocks_on_success_and_keeps_lock_on_error(self, tmp_path: Path) -> None:
        """Context manager leaves ``metadataLock`` active if business logic raises an error."""
        store = _write_test_store(tmp_path / "session.mdio")

        with (
            pytest.raises(RuntimeError, match="polygon failure"),
            z_severance_mask_session(store, "z_severance_failed") as mask_arr,
        ):
            mask_arr[0, 0, 1:3] = True
            msg = "polygon failure"
            raise RuntimeError(msg)

        ds_locked = open_mdio(store)
        with pytest.raises(MDIOSeveranceError, match="metadata lock"):
            resolve_z_severance_mask(ds_locked)

        # Overwrite and complete cleanly
        with z_severance_mask_session(store, "z_severance_failed", overwrite=True) as mask_arr:
            mask_arr[0, 1, 1:3] = True

        ds_ok = open_mdio(store)
        assert resolve_z_severance_mask(ds_ok) == "z_severance_failed"
        assert ds_ok["z_severance_failed"].values[0, 1, 1] is np.True_
        assert ds_ok["z_severance_failed"].values[0, 0, 1] is np.False_

    def test_collision_and_finalize_boundary_validation(self, tmp_path: Path) -> None:
        """Cannot overwrite or finalize non-mask variables even when overwrite=True."""
        store = _write_test_store(tmp_path / "dup.mdio")
        with z_severance_mask_session(store, "mask_a"):
            pass

        with pytest.raises(MDIOAlreadyExistsError):
            create_z_severance_mask(store, "mask_a")

        # Even with overwrite=True, non-mask variables and coordinates are protected
        with pytest.raises(MDIOSeveranceError, match="collides with an existing non-mask"):
            create_z_severance_mask(store, "amplitude", overwrite=True)

        with pytest.raises(MDIOSeveranceError, match="collides with an existing non-mask"):
            create_z_severance_mask(store, "trace_mask", overwrite=True)

        with pytest.raises(MDIOSeveranceError, match="collides with an existing non-mask"):
            create_z_severance_mask(store, "cdp_x", overwrite=True)

        # Finalizing a non-mask variable fails
        with pytest.raises(MDIOSeveranceError, match="not a z-severance mask"):
            finalize_z_severance_mask(store, "amplitude")

        with pytest.raises(MDIOSeveranceError, match="Invalid z-severance mask metadata"):
            create_z_severance_mask(store, "naive_ts", created_on=datetime(2026, 1, 1, 0, 0))  # noqa: DTZ001


class TestResolveAndApplySeveranceMask:
    """Tests for version resolution, explicit selection, and sample zeroing."""

    def test_no_masks_returns_none(self, tmp_path: Path) -> None:
        """Dataset with no severance masks resolves to ``None``."""
        store = _write_test_store(tmp_path / "nomask.mdio")
        ds = open_mdio(store)
        assert resolve_z_severance_mask(ds) is None

    def test_explicit_missing_or_non_mask_variable_raises(self, tmp_path: Path) -> None:
        """Passing a missing or non-mask variable name raises appropriate error."""
        store = _write_test_store(tmp_path / "explicit_missing.mdio")
        ds = open_mdio(store)
        with pytest.raises(MDIOMissingVariableError, match="does_not_exist"):
            resolve_z_severance_mask(ds, severance_mask="does_not_exist")

        with pytest.raises(MDIOSeveranceError, match="not a z-severance mask"):
            resolve_z_severance_mask(ds, severance_mask="trace_mask")

    def test_multiple_versions_selects_latest_chronologically_across_timezones(self, tmp_path: Path) -> None:
        """Default resolution picks the latest ``createdOn`` in UTC regardless of input timezone offset."""
        store = _write_test_store(tmp_path / "versions.mdio")

        # v1: 2026-11-09 18:00 UTC (input +02:00)
        # v2: 2026-11-09 19:00 UTC (input -05:00)
        t_v1 = datetime(2026, 11, 9, 20, 0, 0, tzinfo=timezone(timedelta(hours=2)))
        t_v2 = datetime(2026, 11, 9, 14, 0, 0, tzinfo=timezone(timedelta(hours=-5)))

        with z_severance_mask_session(store, "mask_v1", created_on=t_v1) as m1:
            m1[:, :, 0] = True

        with z_severance_mask_session(store, "mask_v2", created_on=t_v2) as m2:
            m2[:, :, 2:] = True

        ds = open_mdio(store)
        latest_name = resolve_z_severance_mask(ds)
        assert latest_name == "mask_v2"
        assert resolve_z_severance_mask(ds, severance_mask="mask_v1") == "mask_v1"

        # Apply mask_v2 (samples 2 and 3 zeroed)
        ds_latest = apply_z_severance_mask(ds, latest_name)
        assert np.all(ds_latest["amplitude"].values[:, :, 2:] == 0.0)
        assert np.all(ds_latest["amplitude"].values[:, :, :2] != 0.0)

        # Apply mask_v1 (sample 0 zeroed)
        ds_v1 = apply_z_severance_mask(ds, "mask_v1")
        assert np.all(ds_v1["amplitude"].values[:, :, 0] == 0.0)
        assert np.all(ds_v1["amplitude"].values[:, :, 1:] != 0.0)

    def test_tied_timestamps_raises_error(self, tmp_path: Path) -> None:
        """Two masks with identical latest ``createdOn`` raise ``MDIOSeveranceError`` unless specified."""
        store = _write_test_store(tmp_path / "tie.mdio")
        same_ts = datetime(2026, 11, 9, 12, 0, 0, tzinfo=UTC)

        with z_severance_mask_session(store, "mask_1", created_on=same_ts):
            pass
        with z_severance_mask_session(store, "mask_2", created_on=same_ts):
            pass

        ds = open_mdio(store)
        with pytest.raises(MDIOSeveranceError, match="share the same latest"):
            resolve_z_severance_mask(ds)

        # Explicit selection succeeds
        assert resolve_z_severance_mask(ds, severance_mask="mask_1") == "mask_1"

    def test_missing_or_invalid_created_on_raises(self, tmp_path: Path) -> None:
        """A severance mask missing ``createdOn`` raises ``MDIOSeveranceError``."""
        store = _write_test_store(tmp_path / "bad_meta.mdio")
        with z_severance_mask_session(store, "bad_mask"):
            pass

        root = zarr.open_group(store.as_posix(), mode="r+", use_consolidated=False)
        attrs = dict(root["bad_mask"].attrs)
        attrs.pop(CREATED_ON_ATTR, None)
        root["bad_mask"].attrs.clear()
        root["bad_mask"].attrs.update(attrs)

        ds = open_mdio(store)
        with pytest.raises(MDIOSeveranceError, match="has invalid metadata"):
            resolve_z_severance_mask(ds)


class TestMdioToSegyWithZSeverance:
    """End-to-end tests for ``mdio_to_segy`` with z-severance masks."""

    def test_mdio_to_segy_zeroes_severed_samples_and_supports_versions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Export zeroes severed continuous/discrete samples for latest or specified mask version."""
        monkeypatch.setenv("MDIO__IMPORT__SAVE_SEGY_FILE_HEADER", "true")
        segy_in = tmp_path / "input.sgy"
        mdio_path = tmp_path / "survey.mdio"

        spec = _write_tiny_3d_segy(segy_in)
        segy_to_mdio(
            segy_spec=spec,
            mdio_template=TemplateRegistry().get("PostStack3DTime"),
            input_path=segy_in,
            output_path=mdio_path,
            overwrite=True,
        )

        t1 = datetime(2026, 11, 9, 10, 0, 0, tzinfo=UTC)
        t2 = datetime(2026, 11, 9, 11, 0, 0, tzinfo=UTC)

        # v1: sever samples 1..3 across inline=0
        with z_severance_mask_session(mdio_path, "severance_v1", created_on=t1) as mask_v1:
            mask_v1[0, :, 1:3] = True

        # v2 (latest): sever continuous band 2..5 and discrete sample 6 on (inline=0, crossline=0)
        with z_severance_mask_session(mdio_path, "severance_v2", created_on=t2) as mask_v2:
            mask_v2[0, 0, 2:5] = True
            mask_v2[0, 0, 6] = True

        # 1. Default export uses latest (severance_v2)
        out_latest = tmp_path / "export_latest.sgy"
        mdio_to_segy(segy_spec=spec, input_path=mdio_path, output_path=out_latest)
        sgy_latest = SegyFileWrapper(out_latest, spec=spec)
        traces_latest = sgy_latest.trace[:].sample
        # Trace 0 corresponds to (inline=0, crossline=0)
        assert np.all(traces_latest[0, 2:5] == 0.0)
        assert traces_latest[0, 6] == 0.0
        assert np.all(traces_latest[0, :2] == 1.0)
        assert traces_latest[0, 5] == 1.0
        assert traces_latest[0, 7] == 1.0
        # Trace 1 should be untouched in v2
        assert np.all(traces_latest[1] == 2.0)

        # 2. Explicit export of v1
        out_v1 = tmp_path / "export_v1.sgy"
        mdio_to_segy(
            segy_spec=spec,
            input_path=mdio_path,
            output_path=out_v1,
            severance_mask="severance_v1",
        )
        sgy_v1 = SegyFileWrapper(out_v1, spec=spec)
        traces_v1 = sgy_v1.trace[:].sample
        assert np.all(traces_v1[0, 1:3] == 0.0)
        assert traces_v1[0, 0] == 1.0
        assert np.all(traces_v1[0, 3:] == 1.0)

        # 3. Explicit export with non-existent mask raises MDIOMissingVariableError
        with pytest.raises(MDIOMissingVariableError, match="non_existent_mask"):
            mdio_to_segy(
                segy_spec=spec,
                input_path=mdio_path,
                output_path=tmp_path / "should_fail.sgy",
                severance_mask="non_existent_mask",
            )

        # 4. If a new mask is left locked, default mdio_to_segy fails closed,
        #    while explicit export of a completed mask (severance_v1) still succeeds.
        create_z_severance_mask(
            mdio_path,
            "severance_v3_incomplete",
            created_on=datetime(2026, 11, 9, 12, 0, 0, tzinfo=UTC),
        )
        with pytest.raises(MDIOSeveranceError, match="metadata lock"):
            mdio_to_segy(
                segy_spec=spec,
                input_path=mdio_path,
                output_path=tmp_path / "should_fail_locked.sgy",
            )

        out_v1_while_v3_locked = tmp_path / "export_v1_while_v3_locked.sgy"
        mdio_to_segy(
            segy_spec=spec,
            input_path=mdio_path,
            output_path=out_v1_while_v3_locked,
            severance_mask="severance_v1",
        )
        assert out_v1_while_v3_locked.read_bytes() == out_v1.read_bytes()
