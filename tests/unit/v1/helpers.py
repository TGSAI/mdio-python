"""Helper methods used in unit tests."""

from mdio.builder.dataset_builder import MDIODatasetBuilder
from mdio.builder.dataset_builder import _BuilderState
from mdio.builder.dataset_builder import _get_named_dimension
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredField
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.v1.dataset import Dataset
from mdio.builder.schemas.v1.stats import CenteredBinHistogram
from mdio.builder.schemas.v1.stats import SummaryStatistics
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.schemas.v1.units import SpeedUnitEnum
from mdio.builder.schemas.v1.units import SpeedUnitModel
from mdio.builder.schemas.v1.variable import Coordinate
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.schemas.v1.variable import Variable
from mdio.builder.schemas.v1.variable import VariableMetadata


def validate_builder(builder: MDIODatasetBuilder, state: _BuilderState, n_dims: int, n_coords: int, n_var: int) -> None:
    """Validate the state of the builder, the number of dimensions, coordinates, and variables."""
    assert builder._state == state
    assert len(builder._dimensions) == n_dims
    assert len(builder._coordinates) == n_coords
    assert len(builder._variables) == n_var


def validate_coordinate(
    builder: MDIODatasetBuilder, name: str, dims: list[tuple[str, int]], dtype: ScalarType
) -> Coordinate:
    """Validate existence and the structure of the created coordinate."""
    # Validate that coordinate exists
    c = next((c for c in builder._coordinates if c.name == name), None)
    assert c is not None
    assert isinstance(c, Coordinate)

    # Validate that dimensions are stored as NamedDimensions
    for d in dims:
        name = d[0]
        size = d[1]
        assert _get_named_dimension(c.dimensions, name, size) is not None

    assert c.data_type == dtype
    return c


def validate_variable(
    container: MDIODatasetBuilder | Dataset,
    name: str,
    dims: list[tuple[str, int]],
    coords: list[str],
    dtype: ScalarType | StructuredType,
) -> Variable:
    """Validate existence and the structure of the created variable."""
    if isinstance(container, MDIODatasetBuilder):
        var_list = container._variables
        global_coord_list = container._coordinates
    elif isinstance(container, Dataset):
        var_list = container.variables
        global_coord_list = _get_all_coordinates(container)
    else:  # pragma: no cover
        err_msg = f"Expected MDIODatasetBuilder or Dataset, got {type(container)}"
        raise TypeError(err_msg)

    # Validate that the variable exists
    v = next((e for e in var_list if e.name == name), None)
    assert v is not None
    assert isinstance(v, Variable)

    # Validate that dimensions are stored as NamedDimensions within the variable
    assert len(v.dimensions) == len(dims)
    for d in dims:
        name = d[0]
        size = d[1]
        assert _get_named_dimension(v.dimensions, name, size) is not None

    # Validate that coordinates are either embedded or can be resolved from names to Coordinate
    if coords is None:
        assert v.coordinates is None
    else:
        assert len(v.coordinates) == len(coords)
        for coord_name in coords:
            assert _get_coordinate(global_coord_list, v.coordinates, coord_name) is not None

    assert v.data_type == dtype
    return v


def _get_coordinate(
    global_coord_list: list[Coordinate],
    coordinates_or_references: list[Coordinate] | list[str],
    name: str,
) -> Coordinate | None:
    """Get a coordinate by name from the list[Coordinate] | list[str].

    The function validates that the coordinate referenced by the name can be found
    in the global coordinate list.
    If the coordinate is stored as a Coordinate object, it is returned directly.
    """
    if coordinates_or_references is None:  # pragma: no cover
        return None

    for c in coordinates_or_references:
        if isinstance(c, str) and c == name:
            # The coordinate is stored by name (str).
            cc = None
            # Find the Coordinate in the global list and return it.
            if global_coord_list is not None:
                cc = next((cc for cc in global_coord_list if cc.name == name), None)
            if cc is None:  # pragma: no cover
                msg = f"Pre-existing coordinate named {name!r} is not found"
                raise ValueError(msg)
            return cc
        if isinstance(c, Coordinate) and c.name == name:
            # The coordinate is stored as an embedded Coordinate object.
            # Return it.
            return c

    return None  # pragma: no cover


def _get_all_coordinates(dataset: Dataset) -> list[Coordinate]:
    """Get all coordinates from the dataset."""
    all_coords: dict[str, Coordinate] = {}
    for v in dataset.variables:
        if v.coordinates is not None:
            for c in v.coordinates:
                if isinstance(c, Coordinate) and c.name not in all_coords:
                    all_coords[c.name] = c
    return list(all_coords.values())


def make_seismic_poststack_3d_acceptance_dataset(dataset_name: str) -> Dataset:
    """Create in-memory Seismic PostStack 3D Acceptance dataset."""
    ds = MDIODatasetBuilder(
        dataset_name,
        attributes={
            "textHeader": [
                "C01 .......................... ",
                "C02 .......................... ",
                "C03 .......................... ",
            ],
            "foo": "bar",
        },
    )

    # Add dimensions and dimension coordinates
    units_meter = CoordinateMetadata(units_v1=LengthUnitModel(length=LengthUnitEnum.METER))

    ds.add_dimension("inline", 256)
    ds.add_dimension("crossline", 512)
    ds.add_dimension("depth", 384)
    ds.add_coordinate("inline", dimensions=("inline",), data_type=ScalarType.UINT32)
    ds.add_coordinate("crossline", dimensions=("crossline",), data_type=ScalarType.UINT32)
    ds.add_coordinate("depth", dimensions=("depth",), data_type=ScalarType.UINT32, metadata=units_meter)
    # Add regular coordinates
    ds.add_coordinate("cdp_x", dimensions=("inline", "crossline"), data_type=ScalarType.FLOAT32, metadata=units_meter)
    ds.add_coordinate("cdp_y", dimensions=("inline", "crossline"), data_type=ScalarType.FLOAT32, metadata=units_meter)

    chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=(128, 128, 128)))
    common_metadata = VariableMetadata(chunk_grid=chunk_grid)

    # Add image variable
    histogram = CenteredBinHistogram(bin_centers=[1, 2], counts=[10, 15])
    stats = SummaryStatistics(count=100, sum=1215.1, sum_squares=125.12, min=5.61, max=10.84, histogram=histogram)
    image_metadata = common_metadata.model_copy(update={"stats_v1": stats, "attributes": {"fizz": "buzz"}})
    ds.add_variable(
        name="image",
        dimensions=("inline", "crossline", "depth"),
        data_type=ScalarType.FLOAT32,
        compressor=Blosc(cname=BloscCname.zstd),  # also default in zarr3
        coordinates=("cdp_x", "cdp_y"),
        metadata=image_metadata,
    )
    # Add velocity variable
    speed_unit = SpeedUnitModel(speed=SpeedUnitEnum.METER_PER_SECOND)
    velocity_metadata = common_metadata.model_copy(update={"units_v1": speed_unit})
    ds.add_variable(
        name="velocity",
        dimensions=("inline", "crossline", "depth"),
        data_type=ScalarType.FLOAT16,
        coordinates=("cdp_x", "cdp_y"),
        metadata=velocity_metadata,
    )
    # Add inline-optimized image variable
    fast_il_chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=(4, 512, 512)))
    fast_inline_metadata = common_metadata.model_copy(update={"chunk_grid": fast_il_chunk_grid})
    ds.add_variable(
        name="image_inline",
        long_name="inline optimized version of 3d_stack",
        dimensions=("inline", "crossline", "depth"),
        data_type=ScalarType.FLOAT32,
        compressor=Blosc(cname=BloscCname.zstd),  # also default in zarr3
        coordinates=("cdp_x", "cdp_y"),
        metadata=fast_inline_metadata,
    )
    # Add headers variable with structured dtype
    header_chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=(128, 128)))
    header_metadata = VariableMetadata(chunk_grid=header_chunk_grid)
    ds.add_variable(
        name="image_headers",
        dimensions=("inline", "crossline"),
        coordinates=("cdp_x", "cdp_y"),
        data_type=StructuredType(
            fields=[
                StructuredField(name="cdp_x", format=ScalarType.INT32),
                StructuredField(name="cdp_y", format=ScalarType.INT32),
                StructuredField(name="elevation", format=ScalarType.FLOAT16),
                StructuredField(name="some_scalar", format=ScalarType.FLOAT16),
            ]
        ),
        metadata=header_metadata,
    )
    return ds.build()
