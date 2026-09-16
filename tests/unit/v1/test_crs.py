"""Tests for dataset-level coordinate reference system metadata."""

from __future__ import annotations

from mdio.builder.dataset_builder import MDIODatasetBuilder
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.templates.seismic_2d_poststack import Seismic2DPostStackTemplate
from mdio.builder.xarray_builder import to_xarray_dataset


def test_to_xarray_dataset_omits_missing_crs() -> None:
    """Old datasets without a CRS keep the root attribute absent."""
    dataset = (
        MDIODatasetBuilder("test_dataset")
        .add_dimension("inline", 4)
        .add_variable("data", dimensions=("inline",), data_type=ScalarType.FLOAT32)
        .build()
    )
    assert dataset.metadata.crs is None
    assert "crs" not in to_xarray_dataset(dataset).attrs


def test_to_xarray_dataset_includes_crs() -> None:
    """CRS serializes as a root string attribute."""
    dataset = (
        MDIODatasetBuilder("test_dataset", crs="EPSG:32610")
        .add_dimension("inline", 4)
        .add_variable("data", dimensions=("inline",), data_type=ScalarType.FLOAT32)
        .build()
    )
    assert dataset.metadata.crs == "EPSG:32610"
    assert to_xarray_dataset(dataset).attrs["crs"] == "EPSG:32610"


def test_template_crs_forwarded_by_build_dataset() -> None:
    """A configured template CRS is stored on the built dataset and root attrs."""
    template = Seismic2DPostStackTemplate("time")
    template.crs = "EPSG:32610"

    dataset = template.build_dataset("Line", sizes=(4, 8))

    assert dataset.metadata.crs == "EPSG:32610"
    assert to_xarray_dataset(dataset).attrs["crs"] == "EPSG:32610"
