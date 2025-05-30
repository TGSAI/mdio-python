"""Test configuration before everything runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from segy.factory import SegyFactory
from segy.schema import HeaderField
from segy.standards import SegyStandard
from segy.standards import get_segy_standard

from mdio.segy.geometry import StreamerShotGeometryType

if TYPE_CHECKING:
    from pathlib import Path


def create_segy_mock_4d(  # noqa: PLR0913
    fake_segy_tmp: Path,
    num_samples: int,
    shots: list[int],
    cables: list[int],
    receivers_per_cable: list[int],
    guns: list[int] | None = None,
    chan_header_type: StreamerShotGeometryType = StreamerShotGeometryType.A,
    index_receivers: bool = True,
) -> Path:
    """Dummy 4D SEG-Y file for use in tests."""
    segy_path = fake_segy_tmp / f"4d_type_{chan_header_type}.sgy"

    shot_count = len(shots)
    total_chan = np.sum(receivers_per_cable)
    trace_count = shot_count * total_chan

    # Calculate shot, cable, channel/receiver numbers and header values
    cable_headers = []
    channel_headers = []

    for cable, num_rec in zip(cables, receivers_per_cable, strict=True):
        cable_headers.append(np.repeat(cable, num_rec))

        channel_headers.append(np.arange(num_rec) + 1)

    cable_headers = np.hstack(cable_headers)
    channel_headers = np.hstack(channel_headers)

    if chan_header_type == StreamerShotGeometryType.B:
        channel_headers = np.arange(total_chan) + 1

    index_receivers = True
    if chan_header_type == StreamerShotGeometryType.C:
        index_receivers = False

    shot_headers = np.hstack([np.repeat(shot, total_chan) for shot in shots])

    gun_per_shot = [guns[shot % len(guns)] for shot in shots]
    gun_headers = np.hstack([np.repeat(gun, total_chan) for gun in gun_per_shot])

    cable_headers = np.tile(cable_headers, shot_count)
    channel_headers = np.tile(channel_headers, shot_count)

    trace_header_fields = [
        HeaderField(name="field_rec_no", byte=9, format="int32"),
        HeaderField(name="channel", byte=13, format="int32"),
        HeaderField(name="shot_point", byte=17, format="int32"),
        HeaderField(name="offset", byte=37, format="int32"),
        HeaderField(name="samples_per_trace", byte=115, format="int32"),
        HeaderField(name="sample_interval", byte=117, format="int32"),
        HeaderField(name="shot_line", byte=133, format="int16"),
        HeaderField(name="cable", byte=137, format="int16"),
        HeaderField(name="gun", byte=171, format="int16"),
    ]

    rev1_spec = get_segy_standard(1.0)
    spec = rev1_spec.customize(trace_header_fields=trace_header_fields)
    spec.segy_standard = SegyStandard.REV1
    factory = SegyFactory(
        spec=spec,
        sample_interval=1000,
        samples_per_trace=num_samples,
    )

    headers = factory.create_trace_header_template(trace_count)
    samples = factory.create_trace_sample_template(trace_count)

    for trc_idx in range(trace_count):
        shot = shot_headers[trc_idx]
        gun = gun_headers[trc_idx]
        cable = cable_headers[trc_idx]
        channel = channel_headers[trc_idx]
        shot_line = 1
        offset = 0

        if index_receivers is False:
            channel, gun, shot_line = 0, 0, 0

        header_data = (shot, channel, shot, offset, shot_line, cable, gun)

        fields = list(headers.dtype.names)
        fields.remove("samples_per_trace")
        fields.remove("sample_interval")

        headers[fields][trc_idx] = header_data
        samples[trc_idx] = np.linspace(start=shot, stop=shot + 1, num=num_samples)

    with segy_path.open(mode="wb") as fp:
        fp.write(factory.create_textual_header())
        fp.write(factory.create_binary_header())
        fp.write(factory.create_traces(headers, samples))

    return segy_path


@pytest.fixture(scope="module")
def segy_mock_4d_shots(fake_segy_tmp: Path) -> dict[StreamerShotGeometryType, Path]:
    """Generate mock 4D shot SEG-Y files."""
    num_samples = 25
    shots = [2, 3, 5, 6, 7, 8, 9]
    guns = [1, 2]
    cables = [0, 101, 201, 301]
    receivers_per_cable = [1, 5, 7, 5]

    segy_paths = {}

    for chan_header_type in StreamerShotGeometryType:
        segy_paths[chan_header_type] = create_segy_mock_4d(
            fake_segy_tmp,
            num_samples=num_samples,
            shots=shots,
            cables=cables,
            receivers_per_cable=receivers_per_cable,
            chan_header_type=chan_header_type,
            guns=guns,
        )

    return segy_paths
