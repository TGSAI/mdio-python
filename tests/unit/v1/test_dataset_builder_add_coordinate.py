# ruff: noqa: PLR2004
# PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
# The above erroneous warning is generated for every numerical assert.
# Thus, disable it for this file
"""Tests the schema v1 dataset_builder.add_coordinate() public API."""

import pytest

from mdio.schemas.compressors import Blosc
from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.dataset_builder import _BuilderState
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.units import LengthUnitEnum
from mdio.schemas.v1.units import LengthUnitModel
from mdio.schemas.v1.variable import VariableMetadata

from .helpers import validate_builder
from .helpers import validate_coordinate
from .helpers import validate_variable


def test_add_coordinate() -> None:
    """Test adding coordinates. Check the state transition and validate required parameters."""
    builder = MDIODatasetBuilder("test_dataset")
    assert builder._state == _BuilderState.INITIAL

    msg = "Must add at least one dimension before adding coordinates"
    with pytest.raises(ValueError, match=msg):
        builder.add_coordinate(
            "cdp", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT32
        )

    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 200)

    # Validate required parameters
    bad_name = None
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_coordinate(bad_name, dimensions=["speed"], data_type=ScalarType.FLOAT32)
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        builder.add_coordinate("", dimensions=["speed"], data_type=ScalarType.FLOAT32)
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_coordinate("cdp-x", dimensions=None, data_type=ScalarType.FLOAT32)
    with pytest.raises(ValueError, match="'dimensions' must be a non-empty list"):
        builder.add_coordinate("cdp-x", dimensions=[], data_type=ScalarType.FLOAT32)

    # Add a variable using non-existent dimensions
    msg = "Pre-existing dimension named 'xline' is not found"
    with pytest.raises(ValueError, match=msg):
        builder.add_coordinate(
            "bad_cdp-x", dimensions=["inline", "xline"], data_type=ScalarType.FLOAT32
        )

    # Validate state transition
    builder.add_coordinate(
        "cdp-x", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT32
    )
    validate_builder(builder, _BuilderState.HAS_COORDINATES, n_dims=2, n_coords=1, n_var=1)
    validate_variable(
        builder,
        name="cdp-x",
        dims=[("inline", 100), ("crossline", 200)],
        coords=["cdp-x"],
        dtype=ScalarType.FLOAT32,
    )

    # Adding coordinate with the same name twice
    msg = "Adding coordinate with the same name twice is not allowed"
    with pytest.raises(ValueError, match=msg):
        builder.add_coordinate(
            "cdp-x", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT32
        )


def test_add_coordinate_with_defaults() -> None:
    """Test adding coordinates with default arguments."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 200)

    # Add coordinate using defaults
    builder.add_coordinate("cdp", dimensions=["inline", "crossline"], data_type=ScalarType.FLOAT32)
    validate_builder(builder, _BuilderState.HAS_COORDINATES, n_dims=2, n_coords=1, n_var=1)
    validate_coordinate(
        builder, name="cdp", dims=[("inline", 100), ("crossline", 200)], dtype=ScalarType.FLOAT32
    )
    v = validate_variable(
        builder,
        name="cdp",
        dims=[("inline", 100), ("crossline", 200)],
        coords=["cdp"],
        dtype=ScalarType.FLOAT32,
    )
    assert v.long_name == "'cdp' coordinate variable"  # Default value
    assert v.compressor is None  # Default value
    assert v.metadata is None  # Default value


def test_coordinate_with_full_parameters() -> None:
    """Test adding coordinates with all metadata."""
    builder = MDIODatasetBuilder("test_dataset")
    builder.add_dimension("inline", 100)
    builder.add_dimension("crossline", 200)

    # Add coordinate with all metadata
    builder.add_coordinate(
        "cdp",
        long_name="Common Depth Point",
        dimensions=["inline", "crossline"],
        data_type=ScalarType.FLOAT16,
        compressor=Blosc(algorithm="zstd"),
        metadata_info=[
            AllUnits(units_v1=LengthUnitModel(length=LengthUnitEnum.FOOT)),
            UserAttributes(attributes={"MGA": 51, "UnitSystem": "Imperial"}),
        ],
    )
    validate_builder(builder, _BuilderState.HAS_COORDINATES, n_dims=2, n_coords=1, n_var=1)
    c = validate_coordinate(
        builder, name="cdp", dims=[("inline", 100), ("crossline", 200)], dtype=ScalarType.FLOAT16
    )
    assert c.long_name == "Common Depth Point"
    assert isinstance(c.compressor, Blosc)
    assert c.compressor.algorithm == "zstd"
    assert c.metadata.attributes["MGA"] == 51
    assert c.metadata.attributes["UnitSystem"] == "Imperial"
    assert c.metadata.units_v1.length == LengthUnitEnum.FOOT
    v = validate_variable(
        builder,
        name="cdp",
        dims=[("inline", 100), ("crossline", 200)],
        coords=["cdp"],
        dtype=ScalarType.FLOAT16,
    )
    assert isinstance(v.compressor, Blosc)
    assert v.compressor.algorithm == "zstd"
    assert isinstance(v.metadata, VariableMetadata)
    assert v.metadata.units_v1.length == LengthUnitEnum.FOOT
    assert v.metadata.attributes["MGA"] == 51
    assert v.metadata.attributes["UnitSystem"] == "Imperial"
