"""Tests for SEG-Y coordinate extraction and unit resolution helpers."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from xarray import DataArray as xr_DataArray
from xarray import Dataset as xr_Dataset

from mdio.builder.schemas.v1.units import AngleUnitEnum
from mdio.builder.schemas.v1.units import AngleUnitModel
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
from mdio.ingestion.segy.coordinates import _get_coordinates
from mdio.ingestion.segy.coordinates import _get_spatial_coordinate_unit
from mdio.ingestion.segy.coordinates import _populate_coordinates
from mdio.ingestion.segy.coordinates import _update_template_units
from tests.unit.ingestion.testing_helpers import make_grid
from tests.unit.ingestion.testing_helpers import make_grid_with_map
from tests.unit.ingestion.testing_helpers import make_header_array


class TestGetCoordinates:
    """Tests for ``_get_coordinates``."""

    def test_returns_dims_and_coords_in_template_order(self) -> None:
        """Dim coords and non-dim coords should follow the template's declared order."""
        inline = np.array([1, 2, 3], dtype=np.int32)
        crossline = np.array([10, 20], dtype=np.int32)
        sample = np.array([0, 4, 8, 12], dtype=np.int32)
        grid = make_grid([("inline", inline), ("crossline", crossline), ("time", sample)])

        n = inline.size * crossline.size
        cdp_x = np.arange(n, dtype=np.float64)
        cdp_y = np.arange(n, dtype=np.float64) + 100.0
        headers = make_header_array({"cdp_x": cdp_x, "cdp_y": cdp_y})

        template = Seismic3DPostStackTemplate(data_domain="time")

        dim_coords, non_dim = _get_coordinates(grid, headers, template)

        assert [d.name for d in dim_coords] == ["inline", "crossline", "time"]
        np.testing.assert_array_equal(dim_coords[0].coords, inline)
        np.testing.assert_array_equal(dim_coords[1].coords, crossline)
        assert list(non_dim.keys()) == ["cdp_x", "cdp_y"]
        np.testing.assert_array_equal(non_dim["cdp_x"], cdp_x)
        np.testing.assert_array_equal(non_dim["cdp_y"], cdp_y)

    def test_missing_dimension_raises(self) -> None:
        """A template dim missing from the grid should raise ValueError."""
        grid = make_grid(
            [
                ("inline", np.array([1, 2], dtype=np.int32)),
                # Missing 'crossline'
                ("time", np.array([0, 4], dtype=np.int32)),
            ]
        )
        headers = make_header_array({"cdp_x": np.zeros(2, dtype=np.float64), "cdp_y": np.zeros(2, dtype=np.float64)})

        template = Seismic3DPostStackTemplate(data_domain="time")

        with pytest.raises(ValueError, match=r"Dimension 'crossline' was not found"):
            _get_coordinates(grid, headers, template)

    def test_missing_coordinate_field_raises(self) -> None:
        """A template coord absent from SEG-Y headers should raise ValueError."""
        inline = np.array([1, 2], dtype=np.int32)
        crossline = np.array([10, 20], dtype=np.int32)
        sample = np.array([0, 4], dtype=np.int32)
        grid = make_grid([("inline", inline), ("crossline", crossline), ("time", sample)])
        # Headers lack 'cdp_y'
        headers = make_header_array({"cdp_x": np.zeros(4, dtype=np.float64)})

        template = Seismic3DPostStackTemplate(data_domain="time")

        with pytest.raises(ValueError, match=r"Coordinate 'cdp_y' not found"):
            _get_coordinates(grid, headers, template)


class TestPopulateCoordinates:
    """Tests for the ``_populate_coordinates`` wrapper.

    These pin the contract that wraps ``populate_dim_coordinates`` +
    ``populate_non_dim_coordinates``: dim names land in ``drop_vars`` before coord
    names, both halves run, and the wrapper threads its own initially-empty drop
    list.
    """

    def test_wraps_dim_and_non_dim_population_in_order(self) -> None:
        """Wrapper should populate dims then non-dims and concatenate drop lists."""
        inline = np.array([1, 2], dtype=np.int32)
        crossline = np.array([10, 20], dtype=np.int32)
        sample = np.array([0, 4], dtype=np.int32)
        live = [(1, 10), (1, 20), (2, 10), (2, 20)]
        grid = make_grid_with_map(
            [("inline", inline), ("crossline", crossline), ("sample", sample)],
            live_records=live,
        )

        cdp_x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        cdp_x_da = xr_DataArray(np.zeros((2, 2), dtype=np.float64), dims=["inline", "crossline"])
        cdp_x_da.encoding["_FillValue"] = np.float64(-1.0)

        dataset = xr_Dataset(
            {
                "inline": xr_DataArray(np.zeros(2, dtype=np.int32), dims=["inline"]),
                "crossline": xr_DataArray(np.zeros(2, dtype=np.int32), dims=["crossline"]),
                "sample": xr_DataArray(np.zeros(2, dtype=np.int32), dims=["sample"]),
                "cdp_x": cdp_x_da,
            }
        )

        dataset, drop_vars = _populate_coordinates(
            dataset,
            grid,
            coords={"cdp_x": cdp_x},
            spatial_coordinate_scalar=1,
        )

        np.testing.assert_array_equal(dataset["inline"].values, inline)
        np.testing.assert_array_equal(dataset["crossline"].values, crossline)
        np.testing.assert_array_equal(dataset["sample"].values, sample)
        np.testing.assert_array_equal(dataset["cdp_x"].values, cdp_x.reshape((2, 2)))
        # Dim names recorded first, then non-dim coord names.
        assert drop_vars == ["inline", "crossline", "sample", "cdp_x"]


class TestGetSpatialCoordinateUnit:
    """Tests for ``_get_spatial_coordinate_unit``."""

    @pytest.mark.parametrize(
        ("code", "expected_unit"),
        [
            (1, LengthUnitEnum.METER),
            (2, LengthUnitEnum.FOOT),
        ],
    )
    def test_known_measurement_codes(self, code: int, expected_unit: LengthUnitEnum) -> None:
        """Codes 1 (m) and 2 (ft) return the corresponding length unit."""
        info = SimpleNamespace(binary_header_dict={"measurement_system_code": code})
        result = _get_spatial_coordinate_unit(info)
        assert isinstance(result, LengthUnitModel)
        assert result.length == expected_unit

    def test_unknown_code_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unexpected codes should log a warning and return ``None``."""
        info = SimpleNamespace(binary_header_dict={"measurement_system_code": 7})
        with caplog.at_level(logging.WARNING, logger="mdio.ingestion.segy.coordinates"):
            result = _get_spatial_coordinate_unit(info)
        assert result is None
        assert any("Unexpected value in coordinate unit" in r.message for r in caplog.records)


class TestUpdateTemplateUnits:
    """Tests for ``_update_template_units``."""

    def _stub_template(self) -> MagicMock:
        template = MagicMock(spec=AbstractDatasetTemplate)
        template.get_unit_by_key.return_value = None
        return template

    def test_adds_angle_units_only_when_spatial_unit_missing(self) -> None:
        """Without a spatial unit, only angle units should be added."""
        template = self._stub_template()
        result = _update_template_units(template, unit=None)

        template.add_units.assert_called_once()
        added = template.add_units.call_args.args[0]
        assert set(added.keys()) == {"angle", "azimuth"}
        for unit in added.values():
            assert isinstance(unit, AngleUnitModel)
            assert unit.angle == AngleUnitEnum.DEGREES
        assert result is template

    def test_adds_spatial_units_when_unit_provided(self) -> None:
        """A non-None unit should populate all SPATIAL keys plus angle keys."""
        template = self._stub_template()
        unit = LengthUnitModel(length=LengthUnitEnum.METER)

        _update_template_units(template, unit=unit)

        added = template.add_units.call_args.args[0]
        expected_keys = {
            "angle",
            "azimuth",
            "cdp_x",
            "cdp_y",
            "source_coord_x",
            "source_coord_y",
            "group_coord_x",
            "group_coord_y",
            "offset",
        }
        assert set(added.keys()) == expected_keys
        for key in ("cdp_x", "cdp_y", "source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y", "offset"):
            assert added[key] is unit

    def test_preserves_pre_existing_spatial_units(self, caplog: pytest.LogCaptureFixture) -> None:
        """Keys that already have a template unit must not be overwritten."""
        existing = LengthUnitModel(length=LengthUnitEnum.FOOT)
        new_unit = LengthUnitModel(length=LengthUnitEnum.METER)

        template = MagicMock(spec=AbstractDatasetTemplate)

        def fake_lookup(key: str) -> LengthUnitModel | None:
            return existing if key == "cdp_x" else None

        template.get_unit_by_key.side_effect = fake_lookup

        with caplog.at_level(logging.WARNING, logger="mdio.ingestion.segy.coordinates"):
            _update_template_units(template, unit=new_unit)

        added = template.add_units.call_args.args[0]
        assert "cdp_x" not in added
        assert added["cdp_y"] is new_unit
        assert any("already in template" in r.message for r in caplog.records)
