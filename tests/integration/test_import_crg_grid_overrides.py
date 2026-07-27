"""End to end testing for CRG (Continuous Receiver Gathers) SEG-Y to MDIO conversion.

The CRG template declares ``(component, receiver_line, receiver, time)`` and relies on the
``HasDuplicates`` grid override to insert the per-receiver segment axis as a ``trace``
dimension at ingest. These tests exercise the full ``segy_to_mdio`` path for both the
single- and multi-component layouts, proving the template + generalized synthesis hook +
the tuned HasDuplicates chunk/dtype knobs work together.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import dask
import numpy as np
import pytest
import xarray.testing as xrt
from segy.factory import SegyFactory
from segy.schema import HeaderField
from segy.standards import SegyStandard
from segy.standards import get_segy_standard

from mdio.api.io import open_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.converters.segy import segy_to_mdio

if TYPE_CHECKING:
    from pathlib import Path

    from segy.schema import SegySpec

dask.config.set(scheduler="synchronous")
os.environ["MDIO__IMPORT__SAVE_SEGY_FILE_HEADER"] = "true"

CRG_TEMPLATE = "ObnContinuousReceiverGathers3D"


def get_segy_mock_crg_spec(include_component: bool = False) -> SegySpec:
    """Create a mock CRG SEG-Y specification.

    Args:
        include_component: Whether to include a component header field. When omitted, the
            template synthesizes a constant ``component = 1``.

    Returns:
        SegySpec configured for CRG data.
    """
    trace_header_fields = [
        HeaderField(name="orig_field_record_num", byte=9, format="int32"),
        HeaderField(name="channel", byte=13, format="int32"),
        HeaderField(name="samples_per_trace", byte=115, format="int16"),
        HeaderField(name="sample_interval", byte=117, format="int16"),
        HeaderField(name="coordinate_scalar", byte=71, format="int16"),
        HeaderField(name="group_coord_x", byte=81, format="int32"),
        HeaderField(name="group_coord_y", byte=85, format="int32"),
        HeaderField(name="receiver_line", byte=137, format="int16"),
        HeaderField(name="receiver", byte=139, format="int16"),
    ]
    if include_component:
        trace_header_fields.append(HeaderField(name="component", byte=189, format="int16"))

    rev1_spec = get_segy_standard(1.0)
    spec = rev1_spec.customize(trace_header_fields=trace_header_fields)
    spec.segy_standard = SegyStandard.REV1
    return spec


def create_segy_mock_crg(  # noqa: PLR0913
    fake_segy_tmp: Path,
    num_samples: int,
    receiver_line: int,
    receivers: list[int],
    segments_per_receiver: int,
    components: list[int] | None = None,
    filename_suffix: str = "",
) -> Path:
    """Create a mock CRG SEG-Y file.

    Each ``(component, receiver_line, receiver)`` tuple gets ``segments_per_receiver`` traces
    written in acquisition order, so the HasDuplicates counter produces a dense ``trace``
    axis of that length.
    """
    include_component = components is not None
    segy_path = fake_segy_tmp / f"crg{'_' + filename_suffix if filename_suffix else ''}.sgy"

    component_list = components if include_component else [None]
    trace_count = len(component_list) * len(receivers) * segments_per_receiver

    factory = SegyFactory(
        spec=get_segy_mock_crg_spec(include_component=include_component),
        sample_interval=2000,
        samples_per_trace=num_samples,
    )
    headers = factory.create_trace_header_template(trace_count)
    samples = factory.create_trace_sample_template(trace_count)

    start_x = 700000
    start_y = 4000000
    step = 100

    trc_idx = 0
    for component in component_list:
        for receiver_idx, receiver in enumerate(receivers):
            for _segment in range(segments_per_receiver):
                headers["orig_field_record_num"][trc_idx] = receiver
                headers["channel"][trc_idx] = trc_idx + 1
                headers["receiver_line"][trc_idx] = receiver_line
                headers["receiver"][trc_idx] = receiver
                if include_component:
                    headers["component"][trc_idx] = component

                headers["coordinate_scalar"][trc_idx] = -100
                headers["group_coord_x"][trc_idx] = start_x + step * receiver_idx
                headers["group_coord_y"][trc_idx] = start_y + step * receiver_idx

                samples[trc_idx] = np.linspace(start=receiver, stop=receiver + 1, num=num_samples)
                trc_idx += 1

    with segy_path.open(mode="wb") as fp:
        fp.write(factory.create_textual_header())
        fp.write(factory.create_binary_header())
        fp.write(factory.create_traces(headers, samples))

    return segy_path


@pytest.fixture
def segy_mock_crg_with_component(fake_segy_tmp: Path) -> Path:
    """Generate a mock CRG SEG-Y file with a component header (2 components)."""
    return create_segy_mock_crg(
        fake_segy_tmp,
        num_samples=25,
        receiver_line=4871,
        receivers=[5908, 5909, 5910],
        segments_per_receiver=4,
        components=[1, 2],
        filename_suffix="with_component",
    )


@pytest.fixture
def segy_mock_crg_no_component(fake_segy_tmp: Path) -> Path:
    """Generate a mock CRG SEG-Y file without a component header (single component)."""
    return create_segy_mock_crg(
        fake_segy_tmp,
        num_samples=25,
        receiver_line=4871,
        receivers=[5908, 5909, 5910],
        segments_per_receiver=4,
        components=None,
        filename_suffix="no_component",
    )


class TestImportCrgWithComponent:
    """CRG import with an explicit component header and tuned HasDuplicates knobs."""

    def test_import_crg_multicomponent(self, segy_mock_crg_with_component: Path, zarr_tmp: Path) -> None:
        """Multi-component CRG ingests to the expected 5-D layout with a tuned trace chunk."""
        segy_spec = get_segy_mock_crg_spec(include_component=True)
        grid_override = {"HasDuplicates": True, "chunksize": 2, "trace_dtype": "uint32"}

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get(CRG_TEMPLATE),
            input_path=segy_mock_crg_with_component,
            output_path=zarr_tmp,
            overwrite=True,
            grid_overrides=grid_override,
        )

        ds = open_mdio(zarr_tmp)

        assert ds.attrs["attributes"]["gridOverrides"] == grid_override
        # HasDuplicates inserts `trace` between the spatial dims and the vertical axis.
        assert ds["amplitude"].dims == ("component", "receiver_line", "receiver", "trace", "time")
        assert ds.sizes == {"component": 2, "receiver_line": 1, "receiver": 3, "trace": 4, "time": 25}

        xrt.assert_duckarray_equal(ds["component"], [1, 2])
        xrt.assert_duckarray_equal(ds["receiver_line"], [4871])
        xrt.assert_duckarray_equal(ds["receiver"], [5908, 5909, 5910])


class TestImportCrgSyntheticComponent:
    """CRG import without a component header - component is synthesized."""

    def test_import_crg_synthetic_component(self, segy_mock_crg_no_component: Path, zarr_tmp: Path) -> None:
        """Single-component CRG synthesizes component=1 and still builds the trace axis."""
        segy_spec = get_segy_mock_crg_spec(include_component=False)
        grid_override = {"HasDuplicates": True}

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get(CRG_TEMPLATE),
            input_path=segy_mock_crg_no_component,
            output_path=zarr_tmp,
            overwrite=True,
            grid_overrides=grid_override,
        )

        ds = open_mdio(zarr_tmp)

        assert ds["amplitude"].dims == ("component", "receiver_line", "receiver", "trace", "time")
        assert ds.sizes == {"component": 1, "receiver_line": 1, "receiver": 3, "trace": 4, "time": 25}
        # Component synthesized with the default constant value 1.
        xrt.assert_duckarray_equal(ds["component"], [1])
        xrt.assert_duckarray_equal(ds["receiver"], [5908, 5909, 5910])
