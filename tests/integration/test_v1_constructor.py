"""Integration test for MDIO v1 Xarray Zarr constructor."""

from datetime import datetime
from datetime import timezone

import numpy as np
import pytest

from mdio.core.v1 import mdio
from mdio.schema.compressors import ZFP
from mdio.schema.compressors import Blosc
from mdio.schema.dtype import ScalarType
from mdio.schema.dtype import StructuredType
from mdio.core.v1.factory import make_dataset
from mdio.core.v1.factory import make_dataset_metadata
from mdio.core.v1.factory import make_named_dimension
from mdio.core.v1.factory import make_variable
from mdio.core.v1.constructor import write_mdio_metadata


def build_toy_dataset():
    """Build a toy dataset for testing."""
    # core dimensions
    inline = make_named_dimension("inline", 256)
    crossline = make_named_dimension("crossline", 512)
    depth = make_named_dimension("depth", 384)

    # Create dataset metadata
    created = datetime.fromisoformat("2023-12-12T15:02:06.413469-06:00")
    meta = make_dataset_metadata(
        name="campos_3d",
        api_version="1.0.0",
        created_on=created,
        attributes={
            "textHeader": [
                "C01 .......................... ",
                "C02 .......................... ",
                "C03 .......................... ",
            ],
            "foo": "bar",
        },
    )

    # Image variable
    image = make_variable(
        name="image",
        dimensions=[inline, crossline, depth],
        data_type=ScalarType.FLOAT32,
        compressor=Blosc(algorithm="zstd"),
        coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
        metadata={
            "chunkGrid": {
                "name": "regular",
                "configuration": {"chunkShape": [128, 128, 128]},
            },
            "statsV1": {
                "count": 100,
                "sum": 1215.1,
                "sumSquares": 125.12,
                "min": 5.61,
                "max": 10.84,
                "histogram": {"binCenters": [1, 2], "counts": [10, 15]},
            },
            "attributes": {"fizz": "buzz"},
        },
    )

    # Velocity variable
    velocity = make_variable(
        name="velocity",
        dimensions=[inline, crossline, depth],
        data_type=ScalarType.FLOAT16,
        compressor=None,
        coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
        metadata={
            "chunkGrid": {
                "name": "regular",
                "configuration": {"chunkShape": [128, 128, 128]},
            },
            "unitsV1": {"speed": "m/s"},
        },
    )

    # Inline-optimized image variable
    image_inline = make_variable(
        name="image_inline",
        dimensions=[inline, crossline, depth],
        data_type=ScalarType.FLOAT32,
        compressor=ZFP(mode="fixed_accuracy", tolerance=0.05),
        coordinates=["inline", "crossline", "depth", "cdp-x", "cdp-y"],
        metadata={
            "chunkGrid": {
                "name": "regular",
                "configuration": {"chunkShape": [4, 512, 512]},
            }
        },
    )

    # Headers variable with structured dtype
    headers_dtype = StructuredType(
        fields=[
            {"name": "cdp-x", "format": ScalarType.INT32},
            {"name": "cdp-y", "format": ScalarType.INT32},
            {"name": "elevation", "format": ScalarType.FLOAT16},
            {"name": "some_scalar", "format": ScalarType.FLOAT16},
        ]
    )
    image_headers = make_variable(
        name="image_headers",
        dimensions=[inline, crossline],
        data_type=headers_dtype,
        compressor=None,
        coordinates=["inline", "crossline", "cdp-x", "cdp-y"],
        metadata={
            "chunkGrid": {
                "name": "regular",
                "configuration": {"chunkShape": [128, 128]},
            }
        },
    )

    # Standalone dimension variables
    inline_var = make_variable(
        name="inline", dimensions=[inline], data_type=ScalarType.UINT32, compressor=None
    )
    crossline_var = make_variable(
        name="crossline",
        dimensions=[crossline],
        data_type=ScalarType.UINT32,
        compressor=None,
    )
    depth_var = make_variable(
        name="depth",
        dimensions=[depth],
        data_type=ScalarType.UINT32,
        compressor=None,
        metadata={"unitsV1": {"length": "m"}},
    )
    cdp_x = make_variable(
        name="cdp-x",
        dimensions=[inline, crossline],
        data_type=ScalarType.FLOAT32,
        compressor=None,
        metadata={"unitsV1": {"length": "m"}},
    )
    cdp_y = make_variable(
        name="cdp-y",
        dimensions=[inline, crossline],
        data_type=ScalarType.FLOAT32,
        compressor=None,
        metadata={"unitsV1": {"length": "m"}},
    )

    # Compose full dataset
    return make_dataset(
        [
            image,
            velocity,
            image_inline,
            image_headers,
            inline_var,
            crossline_var,
            depth_var,
            cdp_x,
            cdp_y,
        ],
        meta,
    )


def test_to_mdio_writes_and_returns_mdio(tmp_path):
    """Test that to_mdio writes and returns an mdio.Dataset."""
    ds_in = build_toy_dataset()
    store_path = tmp_path / "toy.mdio"
    # write to Zarr and get back xarray.Dataset
    ds_out = write_mdio_metadata(ds_in, str(store_path))
    # global attributes should be present on the returned Dataset
    assert ds_out.attrs["apiVersion"] == ds_in.metadata.api_version
    assert ds_out.attrs["createdOn"] == str(ds_in.metadata.created_on)
    if ds_in.metadata.attributes:
        assert ds_out.attrs["attributes"] == ds_in.metadata.attributes
    # verify the DataArray exists with correct shape and dtype
    arr = ds_out["image"]
    assert arr.shape == (256, 512, 384)
    assert arr.dtype == np.dtype("float32")
