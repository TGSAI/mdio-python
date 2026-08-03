"""End to end testing for continuous receiver gather SEG-Y to MDIO conversion."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import dask
import numpy as np
import pytest
import xarray.testing as xrt
from tests.integration.conftest import crg_expected_epochs
from tests.integration.conftest import get_segy_mock_crg_spec

from mdio.api.io import open_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.converters.segy import segy_to_mdio
from mdio.segy.geometry import GridOverrides

if TYPE_CHECKING:
    from pathlib import Path

    from xarray import Dataset

dask.config.set(scheduler="synchronous")
os.environ["MDIO__IMPORT__SAVE_SEGY_FILE_HEADER"] = "true"

# Mirrors the segy_mock_crg_* fixtures
NUM_SAMPLES = 25
RECEIVER_LINE = 10
RECEIVERS = [101, 102, 103]
COMPONENTS = [1, 2]
NUM_SEGMENTS = 4
TEMPLATE_NAME = "NodalContinuousReceiverGathers3D"
INT64_FILL = np.iinfo(np.int64).max


def _ingest(input_path: Path, output_path: Path, include_component: bool) -> Dataset:
    """Ingest a mock CRG file with CalculateSegmentIndex enabled."""
    segy_to_mdio(
        segy_spec=get_segy_mock_crg_spec(include_component=include_component),
        mdio_template=TemplateRegistry().get(TEMPLATE_NAME),
        input_path=input_path,
        output_path=output_path,
        overwrite=True,
        grid_overrides=GridOverrides(calculate_segment_index=True),
    )
    return open_mdio(output_path)


@pytest.fixture(scope="module")
def crg_dataset(segy_mock_crg_with_component: Path, zarr_tmp: Path) -> Dataset:
    """Ingest the mock CRG deliverable once for structural assertions."""
    return _ingest(segy_mock_crg_with_component, zarr_tmp, include_component=True)


class TestImportContinuousReceiverGathers:
    """End-to-end import of continuous receiver gathers with CalculateSegmentIndex."""

    def test_dimensions(self, crg_dataset: Dataset) -> None:
        """Assert amplitude dimension order and coordinate values."""
        assert crg_dataset["amplitude"].dims == (
            "receiver_line",
            "receiver",
            "component",
            "segment_index",
            "time",
        )

        xrt.assert_duckarray_equal(crg_dataset["component"], COMPONENTS)
        xrt.assert_duckarray_equal(crg_dataset["receiver_line"], [RECEIVER_LINE])
        xrt.assert_duckarray_equal(crg_dataset["receiver"], RECEIVERS)
        xrt.assert_duckarray_equal(crg_dataset["time"], list(range(0, NUM_SAMPLES * 2, 2)))

    def test_segment_index_is_dense_and_positional(self, crg_dataset: Dataset) -> None:
        """segment_index is a dense 0-N axis; no trace dimension is introduced."""
        assert crg_dataset.sizes["segment_index"] == NUM_SEGMENTS
        assert "trace" not in crg_dataset.dims

    def test_grid_is_dense(self, crg_dataset: Dataset) -> None:
        """Every grid cell holds a live trace for the rectangular mock."""
        num_traces = len(COMPONENTS) * len(RECEIVERS) * NUM_SEGMENTS
        assert int(crg_dataset["trace_mask"].sum()) == num_traces
        assert bool(crg_dataset["trace_mask"].all())

    def test_override_recorded_in_metadata(self, crg_dataset: Dataset) -> None:
        """CalculateSegmentIndex is recorded in dataset metadata."""
        assert crg_dataset.attrs["attributes"]["gridOverrides"] == {"CalculateSegmentIndex": True}
        assert crg_dataset["segy_file_header"].attrs["binaryHeader"]["samples_per_trace"] == NUM_SAMPLES

    def test_epoch_preserved_exactly(self, crg_dataset: Dataset) -> None:
        """Epoch values survive unmodified and span the full spatial key."""
        assert "epoch" in crg_dataset.coords
        assert crg_dataset["epoch"].dims == ("receiver_line", "receiver", "component", "segment_index")
        assert crg_dataset["epoch"].dtype == np.int64

        for component in COMPONENTS:
            for receiver_idx, receiver in enumerate(RECEIVERS):
                expected = crg_expected_epochs(receiver_idx, NUM_SEGMENTS)
                actual = crg_dataset["epoch"].sel(
                    component=component,
                    receiver_line=RECEIVER_LINE,
                    receiver=receiver,
                )
                xrt.assert_duckarray_equal(actual, expected)

    def test_segments_ordered_by_epoch_not_file_order(self, crg_dataset: Dataset) -> None:
        """Segments land by epoch order despite shuffled file order."""
        for receiver in RECEIVERS:
            trace = crg_dataset["amplitude"].sel(component=1, receiver_line=RECEIVER_LINE, receiver=receiver)
            recorded = trace.values[:, 0]  # first sample of each segment
            xrt.assert_duckarray_equal(recorded, [receiver * 1000 + s for s in range(NUM_SEGMENTS)])

        epochs = crg_dataset["epoch"].sel(component=1, receiver_line=RECEIVER_LINE).values
        assert np.all(np.diff(epochs, axis=-1) > 0)

    def test_receiver_coordinates_do_not_span_time(self, crg_dataset: Dataset) -> None:
        """Receiver positions are indexed by receiver only."""
        for coord_name in ("group_coord_x", "group_coord_y"):
            assert crg_dataset[coord_name].dims == ("receiver_line", "receiver")

        # Coordinate scalar of -100 divides the stored header values
        xrt.assert_duckarray_equal(crg_dataset["group_coord_x"].squeeze(), [7000.0, 7001.0, 7002.0])
        xrt.assert_duckarray_equal(crg_dataset["group_coord_y"].squeeze(), [40000.0] * len(RECEIVERS))


class TestImportContinuousReceiverGathersSyntheticComponent:
    """Import when the SEG-Y spec omits the component header."""

    def test_import_synthetic_component(self, segy_mock_crg_no_component: Path, zarr_tmp2: Path) -> None:
        """Component is synthesized with constant value 1."""
        ds = _ingest(segy_mock_crg_no_component, zarr_tmp2, include_component=False)

        assert "component" in ds.dims
        xrt.assert_duckarray_equal(ds["component"], [1])
        xrt.assert_duckarray_equal(ds["receiver"], RECEIVERS)
        assert ds.sizes["segment_index"] == NUM_SEGMENTS
        assert bool(ds["trace_mask"].all())


class TestImportContinuousReceiverGathersRagged:
    """Import when receivers have unequal segment counts."""

    def test_ragged_grid_and_epoch_sentinel(self, segy_mock_crg_ragged: Path, tmp_path: Path) -> None:
        """Short receivers leave empty cells filled with the int64 sentinel."""
        ds = _ingest(segy_mock_crg_ragged, tmp_path / "ragged.mdio", include_component=True)

        assert ds.sizes["segment_index"] == 4  # max across receivers
        live = int(ds["trace_mask"].sum())
        cells = int(np.prod([ds.sizes[k] for k in ("receiver_line", "receiver", "component", "segment_index")]))
        assert live == 4 + 3 + 2
        assert live < cells

        # Receiver 103 recorded only 2 segments; later cells are dead with the int64 fill.
        short = ds.sel(component=1, receiver_line=RECEIVER_LINE, receiver=103)
        assert short["trace_mask"].values.tolist() == [True, True, False, False]
        assert int(short["epoch"].values[2]) == INT64_FILL
        assert int(short["epoch"].values[3]) == INT64_FILL
        live_epochs = short["epoch"].values[short["trace_mask"].values]
        np.testing.assert_array_equal(live_epochs, crg_expected_epochs(2, 2))


class TestImportContinuousReceiverGathersUnevenComponents:
    """Import when components of one receiver miss different segments."""

    def test_each_component_indexed_from_its_own_epochs(
        self,
        segy_mock_crg_uneven_components: Path,
        tmp_path: Path,
    ) -> None:
        """Each component ranks independently; missing segments shift only that component."""
        ds = _ingest(segy_mock_crg_uneven_components, tmp_path / "uneven.mdio", include_component=True)

        assert ds.sizes["segment_index"] == 3
        mask = ds["trace_mask"].sel(receiver_line=RECEIVER_LINE, receiver=101)
        assert mask.sel(component=1).values.tolist() == [True, True, True]
        assert mask.sel(component=2).values.tolist() == [True, True, False]

        all_epochs = crg_expected_epochs(0, 3)
        epochs = ds["epoch"].sel(receiver_line=RECEIVER_LINE, receiver=101)
        np.testing.assert_array_equal(epochs.sel(component=1).values, all_epochs)
        # Component 2 holds epochs 0 and 60 packed at index 0 and 1.
        np.testing.assert_array_equal(
            epochs.sel(component=2).values,
            [all_epochs[0], all_epochs[2], INT64_FILL],
        )

        amp = ds["amplitude"].sel(receiver_line=RECEIVER_LINE, receiver=101)
        assert amp.sel(component=1).values[2, 0] == 1000 + 2
        assert amp.sel(component=2).values[1, 0] == 2000 + 2


class TestContinuousReceiverGathersOverrideGuards:
    """Guards for misconfigured continuous receiver gather imports."""

    def test_ingest_without_override_fails(
        self,
        segy_mock_crg_with_component: Path,
        tmp_path: Path,
    ) -> None:
        """Ingestion fails when CalculateSegmentIndex is omitted."""
        with pytest.raises(ValueError, match=r"Required computed fields \['segment_index'\]"):
            segy_to_mdio(
                segy_spec=get_segy_mock_crg_spec(include_component=True),
                mdio_template=TemplateRegistry().get(TEMPLATE_NAME),
                input_path=segy_mock_crg_with_component,
                output_path=tmp_path / "no_override.mdio",
                overwrite=True,
            )
