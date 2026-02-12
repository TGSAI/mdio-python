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

    from segy.schema import SegySpec


def get_segy_mock_4d_spec() -> SegySpec:
    """Create a mock 4D SEG-Y specification."""
    trace_header_fields = [
        HeaderField(name="orig_field_record_num", byte=9, format="int32"),
        HeaderField(name="channel", byte=13, format="int32"),
        HeaderField(name="shot_point", byte=17, format="int32"),
        HeaderField(name="offset", byte=37, format="int32"),
        HeaderField(name="samples_per_trace", byte=115, format="int16"),
        HeaderField(name="sample_interval", byte=117, format="int16"),
        HeaderField(name="sail_line", byte=133, format="int16"),
        HeaderField(name="cable", byte=137, format="int16"),
        HeaderField(name="gun", byte=171, format="int16"),
        HeaderField(name="coordinate_scalar", byte=71, format="int16"),
        HeaderField(name="source_coord_x", byte=73, format="int32"),
        HeaderField(name="source_coord_y", byte=77, format="int32"),
        HeaderField(name="group_coord_x", byte=81, format="int32"),
        HeaderField(name="group_coord_y", byte=85, format="int32"),
        HeaderField(name="cdp_x", byte=181, format="int32"),
        HeaderField(name="cdp_y", byte=185, format="int32"),
    ]
    rev1_spec = get_segy_standard(1.0)
    spec = rev1_spec.customize(trace_header_fields=trace_header_fields)
    spec.segy_standard = SegyStandard.REV1
    return spec


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

    factory = SegyFactory(
        spec=get_segy_mock_4d_spec(),
        sample_interval=1000,
        samples_per_trace=num_samples,
    )

    headers = factory.create_trace_header_template(trace_count)
    samples = factory.create_trace_sample_template(trace_count)

    start_x = 700000
    start_y = 4000000
    step_x = 100
    step_y = 100

    for trc_shot_idx in range(shot_count):
        for trc_chan_idx in range(total_chan):
            trc_idx = trc_shot_idx * total_chan + trc_chan_idx

            shot = shot_headers[trc_idx]
            gun = gun_headers[trc_idx]
            cable = cable_headers[trc_idx]
            channel = channel_headers[trc_idx]
            sail_line = 1
            offset = 0

            if index_receivers is False:
                channel, gun, sail_line = 0, 0, 0

            # Assign dimension coordinate fields with calculated mock data
            header_fields = ["orig_field_record_num", "channel", "shot_point", "offset", "sail_line", "cable", "gun"]
            headers[header_fields][trc_idx] = (shot, channel, shot, offset, sail_line, cable, gun)

            # Assign coordinate fields with mock data
            x = start_x + step_x * trc_shot_idx
            y = start_y + step_y * trc_chan_idx
            headers["coordinate_scalar"][trc_idx] = -100
            coord_fields = ["source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y", "cdp_x", "cdp_y"]
            headers[coord_fields][trc_idx] = (x, y) * 3

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
    cables = [0, 3, 5, 7]
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


def get_segy_mock_obn_spec(include_component: bool = True) -> SegySpec:
    """Create a mock OBN SEG-Y specification.

    Args:
        include_component: Whether to include component header field.

    Returns:
        SegySpec configured for OBN data.
    """
    trace_header_fields = [
        HeaderField(name="orig_field_record_num", byte=9, format="int32"),
        HeaderField(name="receiver", byte=13, format="int32"),
        HeaderField(name="shot_point", byte=17, format="int32"),
        HeaderField(name="samples_per_trace", byte=115, format="int16"),
        HeaderField(name="sample_interval", byte=117, format="int16"),
        HeaderField(name="shot_line", byte=133, format="int16"),
        HeaderField(name="gun", byte=171, format="int16"),
        HeaderField(name="coordinate_scalar", byte=71, format="int16"),
        HeaderField(name="source_coord_x", byte=73, format="int32"),
        HeaderField(name="source_coord_y", byte=77, format="int32"),
        HeaderField(name="group_coord_x", byte=81, format="int32"),
        HeaderField(name="group_coord_y", byte=85, format="int32"),
    ]

    if include_component:
        trace_header_fields.append(HeaderField(name="component", byte=189, format="int16"))

    rev1_spec = get_segy_standard(1.0)
    spec = rev1_spec.customize(trace_header_fields=trace_header_fields)
    spec.segy_standard = SegyStandard.REV1
    return spec


def create_segy_mock_obn(  # noqa: PLR0913
    fake_segy_tmp: Path,
    num_samples: int,
    receivers: list[int],
    shot_lines: list[int],
    guns: list[int],
    shot_points_per_gun: dict[int, list[int]],
    components: list[int] | None = None,
    filename_suffix: str = "",
) -> Path:
    """Create a mock OBN SEG-Y file for use in tests.

    Args:
        fake_segy_tmp: Temporary directory for SEG-Y files.
        num_samples: Number of samples per trace.
        receivers: List of receiver IDs.
        shot_lines: List of shot line IDs.
        guns: List of gun IDs.
        shot_points_per_gun: Dict mapping gun ID to list of shot points for that gun.
        components: List of component IDs. If None, no component header is written.
        filename_suffix: Optional suffix for the filename.

    Returns:
        Path to the created SEG-Y file.
    """
    include_component = components is not None
    segy_path = fake_segy_tmp / f"obn{'_' + filename_suffix if filename_suffix else ''}.sgy"

    # Calculate total trace count
    total_shot_points = sum(len(sps) for sps in shot_points_per_gun.values())
    trace_count = len(receivers) * len(shot_lines) * total_shot_points
    if include_component:
        trace_count *= len(components)

    factory = SegyFactory(
        spec=get_segy_mock_obn_spec(include_component=include_component),
        sample_interval=1000,
        samples_per_trace=num_samples,
    )

    headers = factory.create_trace_header_template(trace_count)
    samples = factory.create_trace_sample_template(trace_count)

    start_x = 700000
    start_y = 4000000
    step_x = 100
    step_y = 100

    trc_idx = 0
    component_list = components if include_component else [None]

    for component in component_list:
        for receiver_idx, receiver in enumerate(receivers):
            for shot_line_idx, shot_line in enumerate(shot_lines):
                for gun in guns:
                    for shot_point in shot_points_per_gun[gun]:
                        # Base header fields
                        headers["orig_field_record_num"][trc_idx] = shot_point
                        headers["receiver"][trc_idx] = receiver
                        headers["shot_point"][trc_idx] = shot_point
                        headers["shot_line"][trc_idx] = shot_line
                        headers["gun"][trc_idx] = gun

                        if include_component:
                            headers["component"][trc_idx] = component

                        # Coordinate fields
                        src_x = start_x + step_x * shot_line_idx
                        src_y = start_y + step_y * shot_point
                        grp_x = start_x + step_x * receiver_idx
                        grp_y = start_y + step_y * receiver_idx

                        headers["coordinate_scalar"][trc_idx] = -100
                        headers["source_coord_x"][trc_idx] = src_x
                        headers["source_coord_y"][trc_idx] = src_y
                        headers["group_coord_x"][trc_idx] = grp_x
                        headers["group_coord_y"][trc_idx] = grp_y

                        # Sample data
                        samples[trc_idx] = np.linspace(
                            start=receiver + shot_point,
                            stop=receiver + shot_point + 1,
                            num=num_samples,
                        )

                        trc_idx += 1

    with segy_path.open(mode="wb") as fp:
        fp.write(factory.create_textual_header())
        fp.write(factory.create_binary_header())
        fp.write(factory.create_traces(headers, samples))

    return segy_path


@pytest.fixture(scope="module")
def segy_mock_obn_with_component(fake_segy_tmp: Path) -> Path:
    """Generate mock OBN SEG-Y file with component header."""
    num_samples = 25
    receivers = [101, 102, 103]
    shot_lines = [1, 2]
    guns = [1, 2]
    components = [1, 2, 3, 4]  # X, Y, Z, Hydrophone

    # Interleaved shot points: gun 1 gets odd, gun 2 gets even
    shot_points_per_gun = {
        1: [1, 3, 5],  # gun 1: odd shot points
        2: [2, 4, 6],  # gun 2: even shot points
    }

    return create_segy_mock_obn(
        fake_segy_tmp,
        num_samples=num_samples,
        receivers=receivers,
        shot_lines=shot_lines,
        guns=guns,
        shot_points_per_gun=shot_points_per_gun,
        components=components,
        filename_suffix="with_component",
    )


@pytest.fixture(scope="module")
def segy_mock_obn_no_component(fake_segy_tmp: Path) -> Path:
    """Generate mock OBN SEG-Y file without component header."""
    num_samples = 25
    receivers = [101, 102, 103]
    shot_lines = [1, 2]
    guns = [1, 2]

    # Interleaved shot points: gun 1 gets odd, gun 2 gets even
    shot_points_per_gun = {
        1: [1, 3, 5],  # gun 1: odd shot points
        2: [2, 4, 6],  # gun 2: even shot points
    }

    return create_segy_mock_obn(
        fake_segy_tmp,
        num_samples=num_samples,
        receivers=receivers,
        shot_lines=shot_lines,
        guns=guns,
        shot_points_per_gun=shot_points_per_gun,
        components=None,  # No component header
        filename_suffix="no_component",
    )
