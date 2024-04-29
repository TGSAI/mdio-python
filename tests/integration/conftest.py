"""Test configuration before everything runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import segyio

from mdio.seismic.geometry import StreamerShotGeometryType


if TYPE_CHECKING:
    from pathlib import Path


def create_segy_mock_4d(  # noqa: PLR0913
    fake_segy_tmp: Path,
    num_samples: int,
    shots: list,
    cables: list,
    receivers_per_cable: list,
    guns: list | None = None,
    chan_header_type: StreamerShotGeometryType = StreamerShotGeometryType.A,
    index_receivers: bool = True,
) -> str:
    """Dummy 4D SEG-Y file for use in tests."""
    spec = segyio.spec()
    segy_file = fake_segy_tmp / f"4d_type_{chan_header_type}.sgy"

    shot_count = len(shots)
    total_chan = np.sum(receivers_per_cable)
    trace_count = shot_count * total_chan

    spec.format = 1
    spec.samples = range(num_samples)
    spec.tracecount = trace_count
    spec.endian = "big"

    # Calculate shot, cable, channel/receiver numbers and header values
    cable_headers = []
    channel_headers = []

    for cable, num_rec in zip(cables, receivers_per_cable):
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

    gun_per_shot = []
    for shot in shots:
        gun_per_shot.append(guns[(shot % len(guns))])
    gun_headers = np.hstack([np.repeat(gun, total_chan) for gun in gun_per_shot])

    cable_headers = np.tile(cable_headers, shot_count)
    channel_headers = np.tile(channel_headers, shot_count)

    with segyio.create(segy_file, spec) as f:
        for trc_idx in range(trace_count):
            shot = shot_headers[trc_idx]
            gun = gun_headers[trc_idx]
            cable = cable_headers[trc_idx]
            channel = channel_headers[trc_idx]
            source_line = 1

            # offset is byte location 37 - offset 4 bytes
            # fldr is byte location 9 - shot 4 byte
            # ep is byte location 17 - shot 4 byte
            # stae is byte location 137 - cable 2 byte
            # tracf is byte location 13 - channel 4 byte
            # grnors is byte location 171 - gun 2 bytes
            # styp is byte location 133 - source_line 2 bytes

            if index_receivers:
                f.header[trc_idx].update(
                    offset=0,
                    fldr=shot,
                    ep=shot,
                    stae=cable,
                    tracf=channel,
                    grnors=gun,
                    styp=source_line,
                )
            else:
                f.header[trc_idx].update(
                    offset=0,
                    fldr=shot,
                    ep=shot,
                    stae=cable,
                )

            samples = np.linspace(start=shot, stop=shot + 1, num=num_samples)
            f.trace[trc_idx] = samples.astype("float32")

        f.bin.update()

    return segy_file


@pytest.fixture(scope="module")
def segy_mock_4d_shots(fake_segy_tmp: str) -> dict[StreamerShotGeometryType, str]:
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
