"""Test module for masked export of n-D SEG-Y files to MDIO.

We procedurally generate n-D SEG-Y files, import them and export both ways with and without
selection masks. We then compare the resulting SEG-Y files to ensure they're identical to
expected full or partial files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from segy import SegyFile
from segy.factory import SegyFactory
from segy.schema import HeaderField
from segy.schema import SegySpec
from segy.standards import get_segy_standard
from tests.conftest import DEBUG_MODE

from mdio import mdio_to_segy
from mdio.api.opener import open_dataset
from mdio.converters.segy import segy_to_mdio
from mdio.core.storage_location import StorageLocation
from mdio.schemas.v1.templates.template_registry import TemplateRegistry

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


@dataclass
class Dimension:
    """Represents a single dimension for a multidimensional grid."""

    name: str
    start: int
    size: int
    step: int


@dataclass
class GridConfig:
    """Represents the configuration for a seismic grid."""

    name: str
    dims: Iterable[Dimension]


@dataclass
class SegyFactoryConfig:
    """Configuration class for SEG-Y creation with SegyFactory."""

    revision: int | float
    header_byte_map: dict[str, int]
    num_samples: int


@dataclass
class SegyToMdioConfig:
    """Configuration class for SEG-Y to MDIO conversion."""

    chunks: Iterable[int]


@dataclass
class SelectionMaskConfig:
    """Configuration class for masking out parts of the grid during export."""

    mask_num_dims: int
    remove_frac: float | int


MaskedExportConfigTypes = GridConfig | SegyFactoryConfig | SegyToMdioConfig | SelectionMaskConfig


@dataclass
class MaskedExportConfig:
    """Configuration class for a masked export test, combining above configs."""

    grid_conf: GridConfig
    segy_factory_conf: SegyFactoryConfig
    segy_to_mdio_conf: SegyToMdioConfig
    selection_conf: SelectionMaskConfig

    def __iter__(self) -> Iterable[MaskedExportConfigTypes]:
        """Allows for unpacking this dataclass in a test."""
        yield self.grid_conf
        yield self.segy_factory_conf
        yield self.segy_to_mdio_conf
        yield self.selection_conf


# fmt: off
STACK_2D_CONF = MaskedExportConfig(
    GridConfig(name="2d_stack", dims=[Dimension("cdp", 1, 1948, 1)]),
    SegyFactoryConfig(revision=1, header_byte_map={"cdp": 21}, num_samples=201),
    SegyToMdioConfig(chunks=[25, 128]),
    SelectionMaskConfig(mask_num_dims=1, remove_frac=0.998),
)

STACK_3D_CONF = MaskedExportConfig(
    GridConfig(name="3d_stack", dims=[Dimension("inline", 10, 20, 1), Dimension("crossline", 100, 20, 2)]),
    SegyFactoryConfig(revision=1, header_byte_map={"inline": 189, "crossline": 193}, num_samples=201),
    SegyToMdioConfig(chunks=[6, 6, 6]),
    SelectionMaskConfig(mask_num_dims=2, remove_frac=0.98),
)

GATHER_2D_CONF = MaskedExportConfig(
    GridConfig(name="2d_gather", dims=[Dimension("cdp", 1, 40, 1), Dimension("offset", 25, 20, 25)]),
    SegyFactoryConfig(revision=1, header_byte_map={"cdp": 21, "offset": 37}, num_samples=201),
    SegyToMdioConfig(chunks=[2, 12, 128]),
    SelectionMaskConfig(mask_num_dims=1, remove_frac=0.9),
)

GATHER_3D_CONF = MaskedExportConfig(
    GridConfig(name="3d_gather", dims=[Dimension("inline", 10, 8, 1), Dimension("crossline", 100, 10, 2), Dimension("offset", 25, 10, 25)]),
    SegyFactoryConfig(revision=1, header_byte_map={"inline": 189, "crossline": 193, "offset": 37}, num_samples=201),
    SegyToMdioConfig(chunks=[4, 4, 2, 128]),
    SelectionMaskConfig(mask_num_dims=2, remove_frac=0.96),
)

STREAMER_2D_CONF = MaskedExportConfig(
    GridConfig(name="2d_streamer", dims=[Dimension("shot_point", 10, 10, 1), Dimension("channel", 25, 60, 25)]),
    SegyFactoryConfig(revision=1, header_byte_map={"shot_point": 7, "channel": 131}, num_samples=201),
    SegyToMdioConfig(chunks=[2, 12, 128]),
    SelectionMaskConfig(mask_num_dims=1, remove_frac=0.7),
)

STREAMER_3D_CONF = MaskedExportConfig(
    GridConfig(name="3d_streamer", dims=[Dimension("shot_point", 10, 5, 1), Dimension("cable", 1, 6, 1), Dimension("channel", 25, 60, 25)]),
    SegyFactoryConfig(revision=1, header_byte_map={"shot_point": 7, "cable": 193, "channel": 131}, num_samples=201),
    SegyToMdioConfig(chunks=[1, 2, 12, 128]),
    SelectionMaskConfig(mask_num_dims=1, remove_frac=0.5),
)

COCA_3D_CONF = MaskedExportConfig(
    GridConfig(name="3d_coca", dims=[Dimension("inline", 10, 8, 1), Dimension("crossline", 100, 8, 2), Dimension("offset", 25, 15, 25), Dimension("azimuth", 0, 4, 30)]),
    SegyFactoryConfig(revision=1, header_byte_map={"inline": 189, "crossline": 193, "offset": 37, "azimuth": 232}, num_samples=201),
    SegyToMdioConfig(chunks=[4, 4, 4, 1, 128]),
    SelectionMaskConfig(mask_num_dims=2, remove_frac=0.9),
)
# fmt: on


def _segy_spec_mock_nd_segy(grid_conf: GridConfig, segy_factory_conf: SegyFactoryConfig) -> SegySpec:
    spec = get_segy_standard(segy_factory_conf.revision)
    header_flds = []
    for dim in grid_conf.dims:
        byte_loc = segy_factory_conf.header_byte_map[dim.name]
        header_flds.append(HeaderField(name=dim.name, byte=byte_loc, format="int32"))

    header_flds.append(HeaderField(name="samples_per_trace", byte=115, format="int16"))
    header_flds.append(HeaderField(name="sample_interval", byte=117, format="int16"))

    # Add coordinates: {SRC-REC-CDP}-X/Y
    header_flds.extend(
        [
            HeaderField(name="coord_scalar", byte=71, format="int16"),
            HeaderField(name="source_coord_x", byte=73, format="int32"),
            HeaderField(name="source_coord_y", byte=77, format="int32"),
            HeaderField(name="group_coord_x", byte=81, format="int32"),
            HeaderField(name="group_coord_y", byte=85, format="int32"),
            HeaderField(name="cdp_x", byte=181, format="int32"),
            HeaderField(name="cdp_y", byte=185, format="int32"),
            # "gun" is not a standard header. Let's put it at ALIAS_FILTER_FREQ / 141
            HeaderField(name="gun", byte=141, format="int16"),
        ]
    )

    spec = spec.customize(trace_header_fields=header_flds)
    spec.segy_standard = segy_factory_conf.revision
    return spec


def mock_nd_segy(path: str, grid_conf: GridConfig, segy_factory_conf: SegyFactoryConfig) -> SegySpec:
    """Create a fake SEG-Y file with a multidimensional grid."""
    spec = _segy_spec_mock_nd_segy(grid_conf, segy_factory_conf)
    factory = SegyFactory(spec=spec, samples_per_trace=segy_factory_conf.num_samples)

    dim_coords = ()
    for dim in grid_conf.dims:
        start, size, step = dim.start, dim.size, dim.step
        stop = start + (size * step)
        dim_coords += (np.arange(start, stop, step),)

    dim_grid = np.meshgrid(*dim_coords, indexing="ij")
    trace_numbers = np.arange(dim_grid[0].size) + 1

    samples = factory.create_trace_sample_template(trace_numbers.size)
    headers = factory.create_trace_header_template(trace_numbers.size)

    # Fill dimension coordinates (e.g. inline, crossline, etc.)
    for dim_idx, dim in enumerate(grid_conf.dims):
        headers[dim.name] = dim_grid[dim_idx].ravel()

    # Fill coordinates (e.g. {SRC-REC-CDP}-X/Y
    headers["coord_scalar"] = -100
    for field in ["cdp_x", "source_coord_x", "group_coord_x"]:
        start = 700000
        step = 100
        stop = start + step * (trace_numbers.size - 0)
        headers[field] = np.arange(start=start, stop=stop, step=step)
    for field in ["cdp_y", "source_coord_y", "group_coord_y"]:
        start = 4000000
        step = 100
        stop = start + step * (trace_numbers.size - 0)
        headers[field] = np.arange(start=start, stop=stop, step=step)

    # Array filled with repeating sequence
    sequence = np.array([1, 2, 3])
    # Calculate the number of times the sequence needs to be repeated
    num_repetitions = int(np.ceil(trace_numbers.size / len(sequence)))
    repeated_array = np.tile(sequence, num_repetitions)[: trace_numbers.size]
    headers["gun"] = repeated_array

    samples[:] = trace_numbers[..., None]

    with fsspec.open(path, mode="wb") as fp:
        fp.write(factory.create_textual_header())
        fp.write(factory.create_binary_header())
        fp.write(factory.create_traces(headers, samples))

    return spec


def generate_selection_mask(selection_conf: SelectionMaskConfig, grid_conf: GridConfig) -> NDArray:
    """Generate a boolean selection mask for a masked export test."""
    spatial_shape = [dim.size for dim in grid_conf.dims]
    mask_dims = selection_conf.mask_num_dims
    mask_dim_shape = [dim.size for dim in grid_conf.dims[:mask_dims]]

    selection_mask = np.zeros(shape=spatial_shape, dtype="bool")
    cut_axes = np.zeros(shape=mask_dim_shape, dtype="bool")

    cut_size = int((1 - selection_conf.remove_frac) * cut_axes.size)
    rand_idx = np.random.choice(cut_axes.size, size=cut_size, replace=False)
    rand_idx = np.unravel_index(rand_idx, mask_dim_shape)
    selection_mask[rand_idx] = True

    return selection_mask


@pytest.fixture
def export_masked_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture that generates temp directory for export tests."""
    if DEBUG_MODE:
        return Path("TMP/export_masked")
    return tmp_path_factory.getbasetemp() / "export_masked"


# fmt: off
@pytest.mark.parametrize(
    "test_conf",
    [STACK_2D_CONF, STACK_3D_CONF, GATHER_2D_CONF, GATHER_3D_CONF, STREAMER_2D_CONF, STREAMER_3D_CONF, COCA_3D_CONF],
    ids=["2d_stack", "3d_stack", "2d_gather", "3d_gather", "2d_streamer", "3d_streamer", "3d_coca"],
)
# fmt: on
class TestNdImportExport:
    """Test import/export of n-D SEG-Ys to MDIO, with and without selection mask."""

    def test_import(self, test_conf: MaskedExportConfig, export_masked_path: Path) -> None:
        """Test import of an n-D SEG-Y file to MDIO.

        NOTE: This test must be executed before running 'test_ingested_mdio', 'test_export', and
        'test_export_masked' tests.
        """
        grid_conf, segy_factory_conf, segy_to_mdio_conf, _ = test_conf

        domain = "Time"
        match grid_conf.name:
            case "2d_stack":
                template_name = "PostStack2D" + domain
            case "3d_stack":
                template_name = "PostStack3D" + domain
            case "2d_gather":
                template_name = "PreStackCdpGathers2D" + domain
            case "3d_gather":
                template_name = "PreStackCdpGathers3D" + domain
            case "2d_streamer":
                template_name = "PreStackShotGathers2D" + domain
            case "3d_streamer":
                template_name = "PreStackShotGathers3D" + domain
            # case "3d_coca":
            #     templateName = "PostStack3D" + domain
            case _:
                err = f"Unsupported test configuration: {grid_conf.name}"
                raise ValueError(err)

        segy_path = export_masked_path / f"{grid_conf.name}.sgy"
        mdio_path = export_masked_path / f"{grid_conf.name}.mdio"

        segy_spec: SegySpec = mock_nd_segy(segy_path, grid_conf, segy_factory_conf)

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get(template_name),
            input_location=StorageLocation(str(segy_path)),
            output_location=StorageLocation(str(mdio_path)),
            overwrite=True,
        )

    def test_ingested_mdio(self, test_conf: MaskedExportConfig, export_masked_path: Path) -> None:
        """Verify if ingested data is correct.

        NOTE: This test must be executed after the 'test_import' successfully completes
        and before running 'test_export' and 'test_export_masked' tests.
        """
        grid_conf, segy_factory_conf, segy_to_mdio_conf, _ = test_conf
        mdio_path = export_masked_path / f"{grid_conf.name}.mdio"

        # Open the MDIO file
        ds = open_dataset(StorageLocation(str(mdio_path)))

        # Test dimensions and ingested dimension headers
        expected_dims = grid_conf.dims
        for expected in expected_dims:
            actual_dim = ds[expected.name]
            assert expected.name == actual_dim.name
            assert expected.size == actual_dim.values.size
            assert expected.start == actual_dim.values[0]

        live_mask = ds["trace_mask"].values

        expected_sizes = [d.size for d in expected_dims]
        num_traces = np.prod(expected_sizes)

        # Ensure live mask is full
        np.testing.assert_equal(live_mask.ravel(), True)

        # Validate trace headers
        headers = ds["headers"]
        assert headers.shape == live_mask.shape
        assert set(headers.dims) == {dim.name for dim in grid_conf.dims}
        # Validate header values
        trace_index = 0
        trace_header = headers.values.flatten()[trace_index]
        expected_x = 700000 + trace_index * 100
        expected_y = 4000000 + trace_index * 100
        expected_gun = 1 + (trace_index % 3)
        assert trace_header["coord_scalar"] == -100
        assert trace_header["source_coord_x"] == expected_x
        assert trace_header["source_coord_y"] == expected_y
        assert trace_header["group_coord_x"] == expected_x
        assert trace_header["group_coord_y"] == expected_y
        assert trace_header["cdp_x"] == expected_x
        assert trace_header["cdp_y"] == expected_y
        assert trace_header["gun"] == expected_gun

        # Validate trace samples
        # Traces have constant samples with the value equal to the trace index
        # Let's get a horizontal slice of the traces at the first sample
        actual = ds["amplitude"].values[..., 0]
        # Create expected array with trace indices
        # The trace index goes from 1 to num_traces
        expected = np.arange(1, num_traces + 1, dtype=np.float32).reshape(live_mask.shape)
        assert np.array_equal(actual, expected)


    def test_export(self, test_conf: MaskedExportConfig, export_masked_path: Path) -> None:
        """Test export of an n-D MDIO file back to SEG-Y.

        NOTE: This test must be executed after the 'test_import' and 'test_ingested_mdio'
        successfully complete.
        """
        grid_conf, segy_factory_conf, segy_to_mdio_conf, _ = test_conf

        segy_path = export_masked_path / f"{grid_conf.name}.sgy"
        mdio_path = export_masked_path / f"{grid_conf.name}.mdio"
        segy_rt_path = export_masked_path / f"{grid_conf.name}_rt.sgy"

        mdio_to_segy(
            segy_spec=_segy_spec_mock_nd_segy(grid_conf, segy_factory_conf),
            input_location=StorageLocation(str(mdio_path)),
            output_location=StorageLocation(str(segy_rt_path))
        )

        expected_sgy = SegyFile(segy_path)
        actual_sgy = SegyFile(segy_rt_path)

        num_traces = expected_sgy.num_traces
        random_indices = np.random.choice(num_traces, 10, replace=False)
        expected_traces = expected_sgy.trace[random_indices]
        actual_traces = actual_sgy.trace[random_indices]

        assert expected_sgy.num_traces == actual_sgy.num_traces
        assert expected_sgy.text_header == actual_sgy.text_header
        assert expected_sgy.binary_header == actual_sgy.binary_header

        # TODO (Dmitriy Repin): Reconcile custom SegySpecs used in the roundtrip SEGY -> MDIO -> SEGY tests
        # https://github.com/TGSAI/mdio-python/issues/610
        assert_array_equal(desired=expected_traces.header, actual=actual_traces.header)
        assert_array_equal(desired=expected_traces.sample, actual=actual_traces.sample)


    def test_export_masked(self, test_conf: MaskedExportConfig, export_masked_path: Path) -> None:
        """Test export of an n-D MDIO file back to SEG-Y with masked export.

        NOTE: This test must be executed after the 'test_import' and 'test_ingested_mdio'
        successfully complete.
        """
        grid_conf, segy_factory_conf, segy_to_mdio_conf, selection_conf = test_conf

        segy_path = export_masked_path / f"{grid_conf.name}.sgy"
        mdio_path = export_masked_path / f"{grid_conf.name}.mdio"
        segy_rt_path = export_masked_path / f"{grid_conf.name}_rt.sgy"

        selection_mask = generate_selection_mask(selection_conf, grid_conf)

        mdio_to_segy(
            segy_spec=_segy_spec_mock_nd_segy(grid_conf, segy_factory_conf),
            input_location=StorageLocation(str(mdio_path)),
            output_location=StorageLocation(str(segy_rt_path)),
            selection_mask=selection_mask
        )

        expected_trc_idx = selection_mask.ravel().nonzero()[0]
        expected_sgy = SegyFile(segy_path)
        actual_sgy = SegyFile(segy_rt_path)

        # TODO (Dmitriy Repin): Reconcile custom SegySpecs used in the roundtrip SEGY -> MDIO -> SEGY tests
        # https://github.com/TGSAI/mdio-python/issues/610
        assert_array_equal(actual_sgy.trace[:].header, expected_sgy.trace[expected_trc_idx].header)
        assert_array_equal(actual_sgy.trace[:].sample, expected_sgy.trace[expected_trc_idx].sample)
