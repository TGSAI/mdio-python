"""Test module for masked export of n-D SEG-Y files to MDIO.

We procedurally generate n-D SEG-Y files, import them and export both ways with and without
selection masks. We then compare the resulting SEG-Y files to ensure they're identical to
expected full or partial files.
"""

from __future__ import annotations

import os
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
from mdio.api.io import open_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.converters.segy import segy_to_mdio

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


os.environ["MDIO__IMPORT__SAVE_SEGY_FILE_HEADER"] = "true"


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

    template: str
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


STACK_2D_CONF = MaskedExportConfig(
    GridConfig(name="2d_stack", dims=[Dimension("cdp", 1, 1948, 1)]),
    SegyFactoryConfig(revision=1, header_byte_map={"cdp": 21}, num_samples=201),
    SegyToMdioConfig(chunks=[25, 128], template="PostStack2DTime"),
    SelectionMaskConfig(mask_num_dims=1, remove_frac=0.998),
)

STACK_3D_CONF = MaskedExportConfig(
    GridConfig(name="3d_stack", dims=[Dimension("inline", 10, 20, 1), Dimension("crossline", 100, 20, 2)]),
    SegyFactoryConfig(revision=1, header_byte_map={"inline": 189, "crossline": 193}, num_samples=201),
    SegyToMdioConfig(chunks=[6, 6, 6], template="PostStack3DTime"),
    SelectionMaskConfig(mask_num_dims=2, remove_frac=0.98),
)

GATHER_2D_CONF = MaskedExportConfig(
    GridConfig(name="2d_gather", dims=[Dimension("cdp", 1, 40, 1), Dimension("offset", 25, 20, 25)]),
    SegyFactoryConfig(revision=1, header_byte_map={"cdp": 21, "offset": 37}, num_samples=201),
    SegyToMdioConfig(chunks=[2, 12, 128], template="PreStackCdpOffsetGathers2DTime"),
    SelectionMaskConfig(mask_num_dims=1, remove_frac=0.9),
)

GATHER_3D_CONF = MaskedExportConfig(
    GridConfig(
        name="3d_gather",
        dims=[Dimension("inline", 10, 8, 1), Dimension("crossline", 100, 10, 2), Dimension("offset", 25, 10, 25)],
    ),
    SegyFactoryConfig(revision=1, header_byte_map={"inline": 189, "crossline": 193, "offset": 37}, num_samples=201),
    SegyToMdioConfig(chunks=[4, 4, 2, 128], template="PreStackCdpOffsetGathers3DTime"),
    SelectionMaskConfig(mask_num_dims=2, remove_frac=0.96),
)

STREAMER_2D_CONF = MaskedExportConfig(
    GridConfig(name="2d_streamer", dims=[Dimension("shot_point", 10, 10, 1), Dimension("channel", 25, 60, 25)]),
    SegyFactoryConfig(revision=1, header_byte_map={"shot_point": 7, "channel": 131}, num_samples=201),
    SegyToMdioConfig(chunks=[2, 12, 128], template="PreStackShotGathers2DTime"),
    SelectionMaskConfig(mask_num_dims=1, remove_frac=0.7),
)

STREAMER_3D_CONF = MaskedExportConfig(
    GridConfig(
        name="3d_streamer",
        dims=[Dimension("shot_point", 10, 5, 1), Dimension("cable", 1, 6, 1), Dimension("channel", 25, 60, 25)],
    ),
    SegyFactoryConfig(revision=1, header_byte_map={"shot_point": 7, "cable": 193, "channel": 131}, num_samples=201),
    SegyToMdioConfig(chunks=[1, 2, 12, 128], template="PreStackShotGathers3DTime"),
    SelectionMaskConfig(mask_num_dims=1, remove_frac=0.5),
)

COCA_3D_CONF = MaskedExportConfig(
    GridConfig(
        name="3d_coca",
        dims=[
            Dimension("inline", 10, 8, 1),
            Dimension("crossline", 100, 8, 2),
            Dimension("offset", 25, 15, 25),
            Dimension("azimuth", 0, 4, 30),
        ],
    ),
    SegyFactoryConfig(
        revision=1, header_byte_map={"inline": 189, "crossline": 193, "offset": 37, "azimuth": 232}, num_samples=201
    ),
    SegyToMdioConfig(chunks=[4, 4, 4, 1, 128], template="PreStackCocaGathers3DTime"),
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
            HeaderField(name="coordinate_scalar", byte=71, format="int16"),
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

    grid_shape = [dim.size for dim in grid_conf.dims]
    grid_unit_spaced = np.mgrid[tuple(slice(None, size) for size in grid_shape)]

    dim_grids = {}
    dim_coords = {}
    for dim, grid in zip(grid_conf.dims, grid_unit_spaced, strict=True):
        dim_grids[dim.name] = grid
        dim_coords[dim.name] = dim.start + grid * dim.step

    num_traces = np.prod(grid_shape)
    trace_numbers = np.arange(num_traces) + 1
    samples = factory.create_trace_sample_template(trace_numbers.size)
    headers = factory.create_trace_header_template(trace_numbers.size)

    # Fill dimension coordinates (e.g. inline, crossline, etc.)
    for key, values in dim_coords.items():
        headers[key] = values.ravel()

    # Fill coordinates (e.g. {SRC-REC-CDP}-X/Y
    headers["coordinate_scalar"] = -100

    x_origin, y_origin = 700_000, 4_000_000
    x_step, y_step = 100, 100

    if grid_conf.name in ("2d_stack", "2d_gather"):
        headers["cdp_x"] = (x_origin + dim_grids["cdp"] * x_step).ravel()
        headers["cdp_y"] = (y_origin + dim_grids["cdp"] * y_step).ravel()
    elif grid_conf.name in ("3d_stack", "3d_gather", "3d_coca"):
        headers["cdp_x"] = (x_origin + dim_grids["inline"] * x_step).ravel()
        headers["cdp_y"] = (y_origin + dim_grids["crossline"] * y_step).ravel()
    elif grid_conf.name in ("2d_streamer", "3d_streamer"):
        headers["source_coord_x"] = (x_origin + dim_grids["shot_point"] * x_step).ravel()
        headers["source_coord_y"] = (y_origin + dim_grids["shot_point"] * y_step).ravel()
        cable_key = "channel" if grid_conf.name == "2d_streamer" else "cable"
        chan_key = "channel" if grid_conf.name == "2d_streamer" else "cable"
        headers["group_coord_x"] = (x_origin + dim_grids["shot_point"] * x_step + dim_grids[chan_key] * x_step).ravel()
        headers["group_coord_y"] = (y_origin + dim_grids["shot_point"] * y_step + dim_grids[cable_key] * y_step).ravel()
        headers["gun"] = np.tile((1, 2, 3), num_traces // 3)

    # for field in ["cdp_x", "source_coord_x", "group_coord_x"]:
    #     start = 700000
    #     step = 100
    #     stop = start + step * (trace_numbers.size - 0)
    #     headers[field] = np.arange(start=start, stop=stop, step=step)
    # for field in ["cdp_y", "source_coord_y", "group_coord_y"]:
    #     start = 4000000
    #     step = 100
    #     stop = start + step * (trace_numbers.size - 0)
    #     headers[field] = np.arange(start=start, stop=stop, step=step)

    samples[:] = trace_numbers[..., None]

    with fsspec.open(path, mode="wb") as fp:
        fp.write(factory.create_textual_header())
        fp.write(factory.create_binary_header())
        fp.write(factory.create_traces(headers, samples))

    return spec


def generate_selection_mask(selection_conf: SelectionMaskConfig, grid_conf: GridConfig) -> NDArray:
    """Generate a boolean selection mask for a masked export test."""
    rng = np.random.default_rng(seed=1234)

    spatial_shape = [dim.size for dim in grid_conf.dims]
    mask_dims = selection_conf.mask_num_dims
    mask_dim_shape = [dim.size for dim in grid_conf.dims[:mask_dims]]

    selection_mask = np.zeros(shape=spatial_shape, dtype="bool")
    cut_axes = np.zeros(shape=mask_dim_shape, dtype="bool")

    cut_size = int((1 - selection_conf.remove_frac) * cut_axes.size)
    rand_idx = rng.choice(cut_axes.size, size=cut_size, replace=False)
    rand_idx = np.unravel_index(rand_idx, mask_dim_shape)
    selection_mask[rand_idx] = True

    return selection_mask


@pytest.fixture
def export_masked_path(tmp_path_factory: pytest.TempPathFactory, raw_headers_env: None) -> Path:  # noqa: ARG001
    """Fixture that generates temp directory for export tests."""
    # Create path suffix based on current raw headers environment variable
    # raw_headers_env dependency ensures the environment variable is set before this runs
    raw_headers_enabled = os.getenv("MDIO__DO_RAW_HEADERS") == "1"
    path_suffix = "with_raw_headers" if raw_headers_enabled else "without_raw_headers"

    if DEBUG_MODE:
        return Path(f"TMP/export_masked_{path_suffix}")
    return tmp_path_factory.getbasetemp() / f"export_masked_{path_suffix}"


@pytest.fixture
def raw_headers_env(request: pytest.FixtureRequest) -> None:
    """Fixture to set/unset MDIO__DO_RAW_HEADERS environment variable."""
    env_value = request.param
    if env_value is not None:
        os.environ["MDIO__DO_RAW_HEADERS"] = env_value
    else:
        os.environ.pop("MDIO__DO_RAW_HEADERS", None)

    yield

    # Cleanup after test - both environment variable and template state
    os.environ.pop("MDIO__DO_RAW_HEADERS", None)

    # Clean up any template modifications to ensure test isolation
    registry = TemplateRegistry.get_instance()

    # Reset any templates that might have been modified with raw headers
    template_names = [
        "PostStack2DTime",
        "PostStack3DTime",
        "PreStackCdpOffsetGathers2DTime",
        "PreStackCdpOffsetGathers3DTime",
        "PreStackShotGathers2DTime",
        "PreStackShotGathers3DTime",
        "PreStackCocaGathers3DTime",
    ]

    for template_name in template_names:
        try:
            template = registry.get(template_name)
            # Remove raw headers enhancement if present
            if hasattr(template, "_mdio_raw_headers_enhanced"):
                delattr(template, "_mdio_raw_headers_enhanced")
                # The enhancement is applied by monkey-patching _add_variables
                # We need to restore it to the original method from the class
                # Since we can't easily restore the exact original, we'll get a fresh instance
                template_class = type(template)
                if hasattr(template_class, "_add_variables"):
                    template._add_variables = template_class._add_variables.__get__(template, template_class)
        except KeyError:
            # Template not found, skip
            continue


@pytest.mark.parametrize(
    "test_conf",
    [STACK_2D_CONF, STACK_3D_CONF, GATHER_2D_CONF, GATHER_3D_CONF, STREAMER_2D_CONF, STREAMER_3D_CONF, COCA_3D_CONF],
    ids=["2d_stack", "3d_stack", "2d_gather", "3d_gather", "2d_streamer", "3d_streamer", "3d_coca"],
)
@pytest.mark.parametrize(
    "raw_headers_env",
    ["1", None],
    ids=["with_raw_headers", "without_raw_headers"],
    indirect=True,
)
class TestNdImportExport:
    """Test import/export of n-D SEG-Ys to MDIO, with and without selection mask."""

    def test_import(self, test_conf: MaskedExportConfig, export_masked_path: Path, raw_headers_env: None) -> None:  # noqa: ARG002
        """Test import of an n-D SEG-Y file to MDIO.

        NOTE: This test must be executed before running 'test_ingested_mdio', 'test_export', and
        'test_export_masked' tests.
        """
        grid_conf, segy_factory_conf, segy_to_mdio_conf, _ = test_conf

        segy_path = export_masked_path / f"{grid_conf.name}.sgy"
        mdio_path = export_masked_path / f"{grid_conf.name}.mdio"

        segy_spec: SegySpec = mock_nd_segy(segy_path, grid_conf, segy_factory_conf)

        template_name = segy_to_mdio_conf.template
        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get(template_name),
            input_path=segy_path,
            output_path=mdio_path,
            overwrite=True,
        )

    def test_ingested_mdio(
        self,
        test_conf: MaskedExportConfig,
        export_masked_path: Path,
        raw_headers_env: None,  # noqa: ARG002
    ) -> None:
        """Verify if ingested data is correct.

        NOTE: This test must be executed after the 'test_import' successfully completes
        and before running 'test_export' and 'test_export_masked' tests.
        """
        grid_conf, segy_factory_conf, segy_to_mdio_conf, _ = test_conf
        mdio_path = export_masked_path / f"{grid_conf.name}.mdio"

        # Open the MDIO file
        ds = open_mdio(mdio_path)

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
        # Validate header values on first trace
        trace_header = headers.values.ravel()[0]
        expected_x = 700_000
        expected_y = 4_000_000
        expected_gun = 1
        assert trace_header["coordinate_scalar"] == -100
        assert trace_header["source_coord_x"] == (expected_x if "streamer" in grid_conf.name else 0)
        assert trace_header["source_coord_y"] == (expected_y if "streamer" in grid_conf.name else 0)
        assert trace_header["group_coord_x"] == (expected_x if "streamer" in grid_conf.name else 0)
        assert trace_header["group_coord_y"] == (expected_y if "streamer" in grid_conf.name else 0)
        assert trace_header["cdp_x"] == (expected_x if "streamer" not in grid_conf.name else 0)
        assert trace_header["cdp_y"] == (expected_y if "streamer" not in grid_conf.name else 0)
        assert trace_header["gun"] == (expected_gun if "streamer" in grid_conf.name else 0)

        # Validate trace samples
        # Traces have constant samples with the value equal to the trace index
        # Let's get a horizontal slice of the traces at the first sample
        actual = ds["amplitude"].values[..., 0]
        # Create expected array with trace indices
        # The trace index goes from 1 to num_traces
        expected = np.arange(1, num_traces + 1, dtype=np.float32).reshape(live_mask.shape)
        assert np.array_equal(actual, expected)

    def test_export(self, test_conf: MaskedExportConfig, export_masked_path: Path, raw_headers_env: None) -> None:  # noqa: ARG002
        """Test export of an n-D MDIO file back to SEG-Y.

        NOTE: This test must be executed after the 'test_import' and 'test_ingested_mdio'
        successfully complete.
        """
        rng = np.random.default_rng(seed=1234)

        grid_conf, segy_factory_conf, segy_to_mdio_conf, _ = test_conf

        segy_path = export_masked_path / f"{grid_conf.name}.sgy"
        mdio_path = export_masked_path / f"{grid_conf.name}.mdio"
        segy_rt_path = export_masked_path / f"{grid_conf.name}_rt.sgy"

        mdio_to_segy(
            segy_spec=_segy_spec_mock_nd_segy(grid_conf, segy_factory_conf),
            input_path=mdio_path,
            output_path=segy_rt_path,
        )

        expected_sgy = SegyFile(segy_path)
        actual_sgy = SegyFile(segy_rt_path)

        num_traces = expected_sgy.num_traces
        random_indices = rng.choice(num_traces, 10, replace=False)
        expected_traces = expected_sgy.trace[random_indices]
        actual_traces = actual_sgy.trace[random_indices]

        assert expected_sgy.num_traces == actual_sgy.num_traces
        assert expected_sgy.text_header == actual_sgy.text_header
        assert expected_sgy.binary_header == actual_sgy.binary_header

        # TODO (Dmitriy Repin): Reconcile custom SegySpecs used in the roundtrip SEGY -> MDIO -> SEGY tests
        # https://github.com/TGSAI/mdio-python/issues/610
        assert_array_equal(desired=expected_traces.header, actual=actual_traces.header)
        assert_array_equal(desired=expected_traces.sample, actual=actual_traces.sample)

    def test_export_masked(
        self,
        test_conf: MaskedExportConfig,
        export_masked_path: Path,
        raw_headers_env: None,  # noqa: ARG002
    ) -> None:
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
            input_path=mdio_path,
            output_path=segy_rt_path,
            selection_mask=selection_mask,
        )

        expected_trc_idx = selection_mask.ravel().nonzero()[0]
        expected_sgy = SegyFile(segy_path)
        actual_sgy = SegyFile(segy_rt_path)

        # TODO (Dmitriy Repin): Reconcile custom SegySpecs used in the roundtrip SEGY -> MDIO -> SEGY tests
        # https://github.com/TGSAI/mdio-python/issues/610
        assert_array_equal(actual_sgy.trace[:].header, expected_sgy.trace[expected_trc_idx].header)
        assert_array_equal(actual_sgy.trace[:].sample, expected_sgy.trace[expected_trc_idx].sample)

    def test_raw_headers_byte_preservation(
        self,
        test_conf: MaskedExportConfig,
        export_masked_path: Path,
        raw_headers_env: None,  # noqa: ARG002
    ) -> None:
        """Test that raw headers are preserved byte-for-byte when MDIO__DO_RAW_HEADERS=1."""
        grid_conf, segy_factory_conf, _, _ = test_conf
        segy_path = export_masked_path / f"{grid_conf.name}.sgy"
        mdio_path = export_masked_path / f"{grid_conf.name}.mdio"

        # Open MDIO dataset
        ds = open_mdio(mdio_path)

        # Check if raw_headers should exist based on environment variable
        has_raw_headers = "raw_headers" in ds.data_vars
        if os.getenv("MDIO__DO_RAW_HEADERS") == "1":
            assert has_raw_headers, "raw_headers should be present when MDIO__DO_RAW_HEADERS=1"
        else:
            assert not has_raw_headers, f"raw_headers should not be present when MDIO__DO_RAW_HEADERS is not set\n {ds}"
            return  # Exit early if raw_headers are not expected

        # Get data (only if raw_headers exist)
        raw_headers_data = ds.raw_headers.values
        trace_mask = ds.trace_mask.values

        # Verify 240-byte headers
        assert raw_headers_data.dtype.itemsize == 240, (
            f"Expected 240-byte headers, got {raw_headers_data.dtype.itemsize}"
        )

        # Read raw bytes directly from SEG-Y file
        def read_segy_trace_header(trace_index: int) -> bytes:
            """Read 240-byte trace header directly from SEG-Y file."""
            # with open(segy_path, "rb") as f:
            with Path.open(segy_path, "rb") as f:
                # Skip text (3200) + binary (400) headers = 3600 bytes
                f.seek(3600)
                # Each trace: 240 byte header + (num_samples * 4) byte samples
                trace_size = 240 + (segy_factory_conf.num_samples * 4)
                trace_offset = trace_index * trace_size
                f.seek(trace_offset, 1)  # Seek relative to current position
                return f.read(240)

        # Compare all valid traces byte-by-byte
        segy_trace_idx = 0
        flat_mask = trace_mask.ravel()
        flat_raw_headers = raw_headers_data.ravel()  # Flatten to 1D array of 240-byte header records

        for grid_idx in range(flat_mask.size):
            if not flat_mask[grid_idx]:
                print(f"Skipping trace {grid_idx} because it is masked")
                continue

            # Get MDIO header as bytes - convert single header record to bytes
            header_record = flat_raw_headers[grid_idx]
            mdio_header_bytes = np.frombuffer(header_record.tobytes(), dtype=np.uint8)

            # Get SEG-Y header as raw bytes directly from file
            segy_raw_header_bytes = read_segy_trace_header(segy_trace_idx)
            segy_header_bytes = np.frombuffer(segy_raw_header_bytes, dtype=np.uint8)

            assert_array_equal(mdio_header_bytes, segy_header_bytes)

            segy_trace_idx += 1
