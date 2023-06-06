"""Test configuration before everything runs."""


from __future__ import annotations

import os

import numpy as np
import pytest
import segyio


def create_segy_mock_4d(
    fake_segy_tmp: str,
    num_samples: int,
    shots: list,
    cables: list,
    receivers_per_cable: list,
    chan_header_type: str = "a",
) -> str:
    """Dummy 4D SEG-Y file for use in tests."""
    spec = segyio.spec()
    segy_file = os.path.join(fake_segy_tmp, f"4d_type_{chan_header_type}.sgy")

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

    # TODO: Add strict=True and remove noqa when minimum Python is 3.10
    for cable, num_rec in zip(cables, receivers_per_cable):  # noqa: B905
        cable_headers.append(np.repeat(cable, num_rec))

        channel_headers.append(np.arange(num_rec) + 1)

    cable_headers = np.hstack(cable_headers)
    channel_headers = np.hstack(channel_headers)

    if chan_header_type == "b":
        channel_headers = np.arange(total_chan) + 1

    shot_headers = np.hstack([np.repeat(shot, total_chan) for shot in shots])
    cable_headers = np.tile(cable_headers, shot_count)
    channel_headers = np.tile(channel_headers, shot_count)

    with segyio.create(segy_file, spec) as f:
        for trc_idx in range(trace_count):
            shot = shot_headers[trc_idx]
            cable = cable_headers[trc_idx]
            channel = channel_headers[trc_idx]

            # offset is byte location 37 - offset 4 bytes
            # fldr is byte location 9 - shot 4 byte
            # ep is byte location 17 - shot 4 byte
            # stae is byte location 137 - cable 2 byte
            # tracf is byte location 13 - channel 4 byte

            f.header[trc_idx].update(
                offset=0,
                fldr=shot,
                ep=shot,
                stae=cable,
                tracf=channel,
            )

            samples = np.linspace(start=shot, stop=shot + 1, num=num_samples)
            f.trace[trc_idx] = samples.astype("float32")

        f.bin.update()

    return segy_file


@pytest.fixture(scope="module")
def segy_mock_4d_shots(fake_segy_tmp: str) -> dict[str, str]:
    """Generate mock 4D shot SEG-Y files."""
    num_samples = 25
    shots = [2, 3, 5]
    cables = [0, 101, 201, 301]
    receivers_per_cable = [1, 5, 7, 5]

    segy_paths = {}

    for type_ in ["a", "b"]:
        segy_paths[type_] = create_segy_mock_4d(
            fake_segy_tmp,
            num_samples=num_samples,
            shots=shots,
            cables=cables,
            receivers_per_cable=receivers_per_cable,
            chan_header_type=type_,
        )

    return segy_paths
