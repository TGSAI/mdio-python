"""Tests for ``read_index_headers``: header subset selection, strategy, dim building."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.templates.types import CoordinateSpec
from mdio.ingestion.schema import DimensionSpec
from mdio.ingestion.schema import ResolvedSchema
from mdio.ingestion.segy import reader
from tests.unit.ingestion.testing_helpers import make_header_array

_READER = "mdio.ingestion.segy.reader"


def _spec_with_fields(*names: str) -> SimpleNamespace:
    """Build a fake SegySpec exposing ``spec.trace.header.fields`` with the given names."""
    fields = [SimpleNamespace(name=name) for name in names]
    return SimpleNamespace(trace=SimpleNamespace(header=SimpleNamespace(fields=fields)))


def _file_info(num_traces: int, sample_labels: np.ndarray) -> SimpleNamespace:
    """Build a fake SegyFileInfo carrying only the attributes the reader touches."""
    return SimpleNamespace(num_traces=num_traces, sample_labels=sample_labels)


def _schema(dimensions: list[DimensionSpec], coordinates: list[CoordinateSpec]) -> ResolvedSchema:
    return ResolvedSchema(
        name="ReaderToy",
        dimensions=dimensions,
        coordinates=coordinates,
        chunk_shape=tuple(-1 for _ in dimensions),
    )


class TestReadIndexHeaders:
    """Tests for the decomposed SEG-Y index-header reader."""

    def test_regular_path_builds_spatial_and_vertical_dims(self) -> None:
        """No overrides: spatial dims come from unique header values, vertical from samples."""
        schema = _schema(
            dimensions=[
                DimensionSpec(name="inline", is_spatial=True),
                DimensionSpec(name="crossline", is_spatial=True),
                DimensionSpec(name="time", is_spatial=False),
            ],
            coordinates=[CoordinateSpec(name="cdp_x", dimensions=("inline", "crossline"), dtype=ScalarType.FLOAT64)],
        )
        headers = make_header_array(
            {
                "inline": np.array([1, 1, 2, 2], dtype=np.int32),
                "crossline": np.array([10, 11, 10, 11], dtype=np.int32),
                "cdp_x": np.array([0.0, 1.0, 2.0, 3.0]),
            }
        )
        segy_file_kwargs = {"spec": _spec_with_fields("inline", "crossline", "cdp_x", "coordinate_scalar")}

        with patch(f"{_READER}.parse_headers", return_value=headers) as mock_parse:
            indexed, dimensions = reader.read_index_headers(
                segy_file_kwargs=segy_file_kwargs,
                file_info=_file_info(4, np.array([0, 2000, 4000])),
                schema=schema,
                grid_overrides=None,
                synthesize_dims=(),
            )

        # Only spec-present required fields are parsed.
        subset = set(mock_parse.call_args.kwargs["subset"])
        assert subset == {"inline", "crossline", "cdp_x", "coordinate_scalar"}

        names = [d.name for d in dimensions]
        assert names == ["inline", "crossline", "time"]
        assert dimensions[0].coords.tolist() == [1, 2]
        assert dimensions[1].coords.tolist() == [10, 11]
        # sample labels normalized by 1000 and cast to int when integral.
        assert dimensions[-1].coords.tolist() == [0, 2, 4]
        assert indexed is headers

    def test_subset_excludes_fields_absent_from_spec(self) -> None:
        """Required fields not in the SEG-Y spec are dropped from the parse subset."""
        schema = _schema(
            dimensions=[
                DimensionSpec(name="inline", is_spatial=True),
                DimensionSpec(name="component", is_spatial=True),
                DimensionSpec(name="time", is_spatial=False),
            ],
            coordinates=[],
        )
        headers = make_header_array({"inline": np.array([1, 2], dtype=np.int32)})
        # spec lacks 'component' on purpose.
        segy_file_kwargs = {"spec": _spec_with_fields("inline")}

        with patch(f"{_READER}.parse_headers", return_value=headers) as mock_parse:
            reader.read_index_headers(
                segy_file_kwargs=segy_file_kwargs,
                file_info=_file_info(2, np.array([0, 1000])),
                schema=schema,
                grid_overrides=None,
                synthesize_dims=(),
            )

        subset = set(mock_parse.call_args.kwargs["subset"])
        assert "component" not in subset
        assert subset == {"inline"}

    def test_synthesize_dims_produces_missing_dimension(self) -> None:
        """A synthesized dim absent from headers is created and yields a dimension."""
        schema = _schema(
            dimensions=[
                DimensionSpec(name="component", is_spatial=True),
                DimensionSpec(name="receiver", is_spatial=True),
                DimensionSpec(name="time", is_spatial=False),
            ],
            coordinates=[],
        )
        headers = make_header_array({"receiver": np.array([5, 6, 7], dtype=np.uint32)})
        segy_file_kwargs = {"spec": _spec_with_fields("receiver")}

        with patch(f"{_READER}.parse_headers", return_value=headers):
            indexed, dimensions = reader.read_index_headers(
                segy_file_kwargs=segy_file_kwargs,
                file_info=_file_info(3, np.array([0, 1000, 2000])),
                schema=schema,
                grid_overrides=None,
                synthesize_dims=("component",),
            )

        names = [d.name for d in dimensions]
        assert names == ["component", "receiver", "time"]
        # ComponentSynthesisStrategy fills the missing dim with a constant 1.
        assert "component" in indexed.dtype.names
        assert dimensions[0].coords.tolist() == [1]
