"""MDIO factories for seismic data. PROTOTYPE."""

from __future__ import annotations

import importlib
import logging
from abc import ABC
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from datetime import UTC
from datetime import datetime
from enum import Enum
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import xarray as xr
import zarr
from dask.array.core import auto_chunks
from segy import SegyFile
from tqdm.auto import tqdm
from xarray import Dataset as xrDataset
from zarr import Group
from zarr import open_group as open_zarr_group
from zarr.storage import FSStore

from mdio.constants import UINT32_MAX
from mdio.core.indexing import ChunkIterator
from mdio.schemas import NamedDimension
from mdio.schemas import ScalarType
from mdio.schemas import StructuredType
from mdio.schemas.builder import DatasetBuilder
from mdio.schemas.builder import VariableBuilder
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset import DatasetMetadata
from mdio.schemas.v1.stats import CenteredBinHistogram
from mdio.schemas.v1.stats import StatisticsMetadata
from mdio.schemas.v1.stats import SummaryStatistics
from mdio.seismic.utilities import segy_export_rechunker


if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from numpy.typing import DTypeLike
    from segy.arrays import HeaderArray

    from mdio.schemas.v1.variable import Variable


logger = logging.getLogger(__name__)


MDIO_VERSION = importlib.metadata.version("multidimio")


class MDIOSchemaType(Enum):
    """MDIO templates for specific data types."""

    SEISMIC_3D_POST_STACK_TIME = auto()
    SEISMIC_3D_POST_STACK_DEPTH = auto()
    SEISMIC_3D_PRE_STACK_CDP_TIME = auto()
    SEISMIC_3D_PRE_STACK_CDP_TIME_IRREGULAR = auto()
    SEISMIC_3D_PRE_STACK_CDP_DEPTH = auto()
    SEISMIC_3D_PRE_STACK_CDP_DEPTH_IRREGULAR = auto()
    SEISMIC_3D_STREAMER_SHOT = auto()

    SEISMIC_2D_POST_STACK_TIME = auto()
    SEISMIC_2D_POST_STACK_DEPTH = auto()
    SEISMIC_2D_PRE_STACK_CDP_TIME = auto()
    SEISMIC_2D_PRE_STACK_CDP_DEPTH = auto()
    SEISMIC_2D_STREAMER_SHOT = auto()

    WIND_WRF = auto()


def get_approx_chunks(
    shape: list[int],
    dtype: DTypeLike,
    limit: str = "4M",
) -> tuple[int, ...]:
    """Get approximate chunk sizes to fit within limit with shape aspect ratio."""
    n_dim = len(shape)
    return auto_chunks(
        chunks=("auto",) * n_dim,
        shape=shape,
        limit=limit,
        dtype=np.dtype(dtype),
        previous_chunks=shape,
    )


class AbstractSeismic(ABC):
    """Abstract class for specific seismic schemas."""

    _trace_domain: str = "unknown"
    _sample_format: str = "float32"
    _dim_names: list[str] = []
    _chunks: list[int] = []
    _coords: dict[str, tuple[str, tuple[int, ...], dict[str, str]]]
    _dataset_attrs: dict[str, str]
    _sample_compressor: dict[str, Any] | None = {"name": "blosc", "algorithm": "zstd"}
    _meta_compressor: dict[str, Any] | None = {"name": "blosc"}

    @classmethod
    def create_dimension_coords(
        cls: type[AbstractSeismic],
        shape: list[int],
        z_units: dict[str, str],
    ) -> list[Variable]:
        """Create schema for dimension coordinates."""
        dim_coords = []
        for dim_name, dim_size in zip(cls._dim_names, shape):
            dim_builder = VariableBuilder()
            dim_builder.set_name(dim_name)
            dim_builder.set_format("uint16")
            dim_builder.add_dimension({dim_name: dim_size})
            dim_builder.set_compressor(cls._meta_compressor)
            if dim_name in cls._trace_domain:
                dim_builder.set_units(z_units)

            dim_coord = dim_builder.build()
            dim_coords.append(dim_coord)

        return dim_coords

    @classmethod
    def create_seismic_variables(  # noqa: PLR0913
        cls: type[AbstractSeismic],
        sample_format: ScalarType,
        header_fields: dict[str, Any],
        shape: list[int],
        chunks: list[int],
        sample_units: dict[str, str],
        coord_names: list[str],
    ) -> tuple[Variable, Variable, Variable]:
        """Build seismic variables based on user input."""
        mask_chunks = list(get_approx_chunks(shape[:-1], "bool", limit="8M"))

        mask_builder = VariableBuilder()
        mask_builder.set_name("trace_mask")
        mask_builder.set_format("bool")
        mask_builder.set_chunks(mask_chunks)
        mask_builder.add_dimension(*cls._dim_names[:-1])
        mask_builder.set_compressor(cls._meta_compressor)

        sample_builder = VariableBuilder()
        sample_builder.set_name("seismic")
        sample_builder.set_format(sample_format)
        sample_builder.set_chunks(chunks)
        sample_builder.set_units(sample_units)
        sample_builder.add_dimension(*cls._dim_names)
        sample_builder.set_compressor(cls._sample_compressor)
        sample_builder.add_coordinate("trace_mask")

        header_builder = VariableBuilder()
        header_builder.set_name("headers")
        header_builder.set_format(header_fields)
        header_builder.set_chunks(chunks[:-1])
        header_builder.add_dimension(*cls._dim_names[:-1])
        header_builder.set_compressor(cls._meta_compressor)
        header_builder.add_coordinate("trace_mask")

        if coord_names is not None:
            mask_builder.add_coordinate(*coord_names)
            sample_builder.add_coordinate(*coord_names)
            header_builder.add_coordinate(*coord_names)

        trace_mask = mask_builder.build()
        samples = sample_builder.build()
        headers = header_builder.build()

        return trace_mask, samples, headers

    @classmethod
    def create_seismic_coordinates(
        cls: type[AbstractSeismic],
        coords_dict: dict[str, tuple[str, dict[str, str], list[str]]],
        shape: list[int],
    ) -> list[Variable]:
        """Build seismic coordinates based on user input."""
        coord_vars = []
        for name, (format_, unit, coord_dims) in coords_dict.items():
            dim_indices = [cls._dim_names.index(dim) for dim in coord_dims]
            coord_shape = [shape[idx] for idx in dim_indices]

            coord_chunks = list(get_approx_chunks(coord_shape, format_, limit="8M"))

            coord_builder = VariableBuilder()
            coord_builder.set_name(name)
            coord_builder.set_format(format_)
            coord_builder.set_chunks(coord_chunks)
            coord_builder.set_units(unit)
            coord_builder.add_dimension(*coord_dims)
            coord_builder.set_compressor(cls._meta_compressor)

            coord_vars.append(coord_builder.build())

        return coord_vars

    @classmethod
    def create(  # noqa: PLR0913
        cls: type[AbstractSeismic],
        name: str,
        shape: list[int],
        header_fields: dict[str, str],
        create_coords: bool = False,
        sample_format: str | None = None,
        chunks: list[int] | None = None,
        sample_units: dict[str, str] | None = None,
        z_units: dict[str, str] | None = None,
    ) -> Dataset:
        """Create a seismic dataset schema based on user input."""
        chunks = chunks or cls._chunks
        sample_format = sample_format or cls._sample_format

        n_dim = len(cls._dim_names)

        if len(shape) != n_dim:
            msg = f"Shape must be {n_dim} dimensional but got {shape}."
            raise ValueError(msg)

        if len(chunks) != n_dim:
            msg = f"Chunks must be {n_dim} dimensional but got {chunks}."
            raise ValueError(msg)

        dim_coords = cls.create_dimension_coords(shape, z_units)
        dataset_vars = dim_coords

        coord_names = None
        if create_coords:
            coord_names = list(cls._coords.keys())
            coord_vars = cls.create_seismic_coordinates(cls._coords, shape)
            dataset_vars += coord_vars

        trace_mask, samples, headers = cls.create_seismic_variables(
            sample_format,
            header_fields,
            shape,
            chunks,
            sample_units,
            coord_names,
        )

        dataset_vars += [trace_mask, samples, headers]

        dataset_meta = DatasetMetadata(
            name=name,
            created_on=datetime.now(UTC).isoformat(),
            api_version=MDIO_VERSION,
        )

        dataset_meta.attributes = cls._dataset_attrs

        dataset_builder = DatasetBuilder()
        dataset_builder.set_name(name)

        return Dataset(variables=dataset_vars, metadata=dataset_meta)


class Seismic3DPostStackTime(AbstractSeismic):
    """3D seismic post stack in time domain."""

    _dataset_attrs = {
        "surveyDimensionality": "3D",
        "ensembleType": "line",
        "processingStage": "post-stack",
    }
    _trace_domain = "time"
    _dim_names = ["inline", "crossline", _trace_domain]
    _chunks = [128, 128, 128]  # 8 mb
    _coords = {
        "cdp-x": ("float64", {"length": "m"}, _dim_names[:-1]),
        "cdp-y": ("float64", {"length": "m"}, _dim_names[:-1]),
    }


class Seismic3DPostStackDepth(AbstractSeismic):
    """3D seismic post stack in depth domain."""

    _dataset_attrs = {
        "surveyDimensionality": "3D",
        "ensembleType": "line",
        "processingStage": "post-stack",
    }
    _trace_domain = "depth"
    _dim_names = ["inline", "crossline", _trace_domain]
    _chunks = [128, 128, 128]  # 8 mb
    _coords = {
        "cdp-x": ("float64", {"length": "m"}, _dim_names[:-1]),
        "cdp-y": ("float64", {"length": "m"}, _dim_names[:-1]),
    }


class Seismic3DPreStackCdpTime(AbstractSeismic):
    """3D seismic CDP gathers in time domain."""

    _dataset_attrs = {
        "surveyDimensionality": "3D",
        "ensembleType": "cdp",
        "processingStage": "pre-stack",
    }
    _trace_domain = "time"
    _dim_names = ["inline", "crossline", "offset", _trace_domain]
    _chunks = [1, 1, 512, 4096]  # 8 mb
    _coords = {
        "cdp-x": ("float64", {"length": "m"}, _dim_names[:-2]),
        "cdp-y": ("float64", {"length": "m"}, _dim_names[:-2]),
    }


class Seismic3DPreStackCdpTimeIrregular(AbstractSeismic):
    """3D seismic CDP gathers in time domain with non-regularized offsets."""

    _dataset_attrs = {
        "surveyDimensionality": "3D",
        "ensembleType": "cdp",
        "processingStage": "pre-stack",
    }
    _trace_domain = "time"
    _dim_names = ["inline", "crossline", "trace", _trace_domain]
    _chunks = [1, 1, 512, 4096]  # 8 mb
    _coords = {
        "cdp-x": ("float64", {"length": "m"}, _dim_names[:-2]),
        "cdp-y": ("float64", {"length": "m"}, _dim_names[:-2]),
        "offset": ("float32", {"length": "m"}, _dim_names[:-1]),
    }


class Seismic3DPreStackCdpDepth(AbstractSeismic):
    """3D seismic CDP gathers in depth domain."""

    _dataset_attrs = {
        "surveyDimensionality": "3D",
        "ensembleType": "cdp",
        "processingStage": "pre-stack",
    }
    _trace_domain = "depth"
    _dim_names = ["inline", "crossline", "offset", _trace_domain]
    _chunks = [1, 1, 512, 4096]  # 8 mb
    _coords = {
        "cdp-x": ("float64", {"length": "m"}, _dim_names[:-2]),
        "cdp-y": ("float64", {"length": "m"}, _dim_names[:-2]),
    }


class Seismic3DPreStackCdpDepthIrregular(AbstractSeismic):
    """3D seismic CDP gathers in depth domain with non-regularized offsets."""

    _dataset_attrs = {
        "surveyDimensionality": "3D",
        "ensembleType": "cdp",
        "processingStage": "pre-stack",
    }
    _trace_domain = "depth"
    _dim_names = ["inline", "crossline", "trace", _trace_domain]
    _chunks = [1, 1, 512, 4096]  # 8 mb
    _coords = {
        "cdp-x": ("float64", {"length": "m"}, _dim_names[:-2]),
        "cdp-y": ("float64", {"length": "m"}, _dim_names[:-2]),
        "offset": ("float32", {"length": "m"}, _dim_names[:-1]),
    }


class Seismic3DStreamerShot(AbstractSeismic):
    """3D seismic shot gathers for streamer acquisition."""

    _dataset_attrs = {
        "surveyDimensionality": "3D",
        "ensembleType": "shot",
        "processingStage": "pre-stack",
    }
    _trace_domain = "time"
    _dim_names = ["shot_point", "cable", "channel", _trace_domain]
    _chunks = [1, 1, 128, 4096]  # 2 mb
    _coords = {
        "gun": ("uint8", None, _dim_names[:-3]),
        "shot-x": ("float64", {"length": "m"}, _dim_names[:-3]),
        "shot-y": ("float64", {"length": "m"}, _dim_names[:-3]),
        "receiver-x": ("float64", {"length": "m"}, _dim_names[:-1]),
        "receiver-y": ("float64", {"length": "m"}, _dim_names[:-1]),
    }


class Seismic2DPostStackTime(AbstractSeismic):
    """2D seismic post stack in time domain."""

    _dataset_attrs = {
        "surveyDimensionality": "2D",
        "ensembleType": "line",
        "processingStage": "post-stack",
    }
    _trace_domain = "time"
    _dim_names = ["cdp", _trace_domain]
    _chunks = [512, 2048]  # 4 mb
    _coords = {
        "cdp-x": ("float64", {"length": "m"}, _dim_names[:-1]),
        "cdp-y": ("float64", {"length": "m"}, _dim_names[:-1]),
    }


class Seismic2DPostStackDepth(AbstractSeismic):
    """2D seismic post stack in depth domain."""

    _dataset_attrs = {
        "surveyDimensionality": "2D",
        "ensembleType": "line",
        "processingStage": "post-stack",
    }
    _trace_domain = "depth"
    _dim_names = ["cdp", _trace_domain]
    _chunks = [512, 2048]  # 4 mb
    _coords = {
        "cdp-x": ("float64", {"length": "m"}, _dim_names[:-1]),
        "cdp-y": ("float64", {"length": "m"}, _dim_names[:-1]),
    }


class Seismic2DPreStackCdpTime(AbstractSeismic):
    """2D seismic CDP gathers in time domain."""

    _dataset_attrs = {
        "surveyDimensionality": "2D",
        "ensembleType": "cdp",
        "processingStage": "pre-stack",
    }
    _trace_domain = "time"
    _dim_names = ["cdp", "offset", _trace_domain]
    _chunks = [1, 512, 2048]  # 4 mb
    _coords = {
        "cdp-x": ("float32", {"length": "m"}, _dim_names[:-2]),
        "cdp-y": ("float32", {"length": "m"}, _dim_names[:-2]),
    }


class Seismic2DPreStackCdpDepth(AbstractSeismic):
    """2D seismic CDP gathers in depth domain."""

    _dataset_attrs = {
        "surveyDimensionality": "2D",
        "ensembleType": "cdp",
        "processingStage": "pre-stack",
    }
    _trace_domain = "depth"
    _dim_names = ["cdp", "offset", _trace_domain]
    _chunks = [1, 512, 2048]  # 4 mb
    _coords = {
        "cdp-x": ("float64", {"length": "m"}, _dim_names[:-2]),
        "cdp-y": ("float64", {"length": "m"}, _dim_names[:-2]),
    }


class Seismic2DStreamerShot(AbstractSeismic):
    """2D seismic shot gathers for streamer acquisition."""

    _dataset_attrs = {
        "surveyDimensionality": "2D",
        "ensembleType": "shot",
        "processingStage": "pre-stack",
    }
    _trace_domain = "time"
    _dim_names = ["shot_point", "channel", _trace_domain]
    _chunks = [1, 128, 4096]
    _coords = {
        "gun": ("uint8", None, _dim_names[:-2]),
        "shot-x": ("float64", {"length": "m"}, _dim_names[:-2]),
        "shot-y": ("float64", {"length": "m"}, _dim_names[:-2]),
        "receiver-x": ("float64", {"length": "m"}, _dim_names[:-1]),
        "receiver-y": ("float64", {"length": "m"}, _dim_names[:-1]),
    }


SCHEMA_TEMPLATE_MAP = {
    # 3D Seismic Post Stack
    MDIOSchemaType.SEISMIC_3D_POST_STACK_TIME: Seismic3DPostStackTime,
    MDIOSchemaType.SEISMIC_3D_POST_STACK_DEPTH: Seismic3DPostStackDepth,
    # 3D Seismic Pre-Stack
    MDIOSchemaType.SEISMIC_3D_PRE_STACK_CDP_TIME: Seismic3DPreStackCdpTime,
    MDIOSchemaType.SEISMIC_3D_PRE_STACK_CDP_TIME_IRREGULAR: Seismic3DPreStackCdpTimeIrregular,
    MDIOSchemaType.SEISMIC_3D_PRE_STACK_CDP_DEPTH: Seismic3DPreStackCdpDepth,
    MDIOSchemaType.SEISMIC_3D_PRE_STACK_CDP_DEPTH_IRREGULAR: Seismic3DPreStackCdpDepthIrregular,
    # 3D Seismic Shot
    MDIOSchemaType.SEISMIC_3D_STREAMER_SHOT: Seismic3DStreamerShot,
    # 2D Seismic Post Stack
    MDIOSchemaType.SEISMIC_2D_POST_STACK_TIME: Seismic2DPostStackTime,
    MDIOSchemaType.SEISMIC_2D_POST_STACK_DEPTH: Seismic2DPostStackDepth,
    # 2D Seismic Pre-Stack
    MDIOSchemaType.SEISMIC_2D_PRE_STACK_CDP_TIME: Seismic2DPostStackTime,
    MDIOSchemaType.SEISMIC_2D_PRE_STACK_CDP_DEPTH: Seismic2DPostStackDepth,
    # 2D Seismic Shot
    MDIOSchemaType.SEISMIC_2D_STREAMER_SHOT: Seismic2DStreamerShot,
}

fill_value_map = {
    "bool": None,
    "float16": np.nan,
    "float32": np.nan,
    "float64": np.nan,
    "uint8": 2**8 - 1,
    "uint16": 2**16 - 1,
    "uint32": 2**32 - 1,
    "uint64": 2**64 - 1,
    "int8": 2**8 // 2 - 1,
    "int16": 2**16 // 2 - 1,
    "int32": 2**32 // 2 - 1,
    "int64": 2**64 // 2 - 1,
}

model_dump_kwargs = {
    "mode": "json",
    "exclude_unset": True,
    "exclude_defaults": True,
    "exclude_none": True,
    "by_alias": True,
}


class MDIOFactory:
    """Factory class for creating MDIO datasets."""

    _schema: Dataset | None = None
    _store: FSStore | None = None
    _root: Group | None = None
    _dump_kw = {"exclude_none": True, "exclude_unset": True, "by_alias": True}

    def __init__(self, uri: str, kind: MDIOSchemaType) -> None:
        self.uri = uri
        self.kind = kind
        self._schema_maker = SCHEMA_TEMPLATE_MAP[self.kind]

    def _ensure_schema(self) -> None:
        """Check if schema is built."""
        if self._schema is None:
            msg = "No schema generated. Call `generate_dataset_schema` first."
            raise ValueError(msg)

    @property
    def schema_json(self) -> str:
        """Return JSON string of the schema."""
        self._ensure_schema()
        return self._schema.model_dump_json(indent=2, **self._dump_kw)

    @property
    def schema(self) -> dict[str, Any]:
        """Return dict representation of the schema."""
        self._ensure_schema()
        return self._schema.model_dump(mode="json", **self._dump_kw)

    def create_schema(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Create the schema from factory."""
        self._schema = self._schema_maker.create(*args, **kwargs)

    def make_mdio_dataset(self) -> None:  # noqa: PLR0912, PLR0915
        """Make MDIO dataset from schema."""
        self._ensure_schema()

        schema = self._schema.model_copy(deep=True)

        self._store = FSStore(str(self.uri))
        self._root = open_zarr_group(self._store, mode="w")

        dataset_metadata = schema.metadata
        if dataset_metadata is not None:
            meta_dict = dataset_metadata.model_dump(**model_dump_kwargs)
            self._root.attrs.update(meta_dict)

        all_dims = {}
        dim_coord_vars = []
        for variable in schema.variables:
            dims = variable.dimensions

            dim_names = []
            for dim in dims:
                if isinstance(dim, NamedDimension):
                    dim_names.append(dim.name)
                    all_dims[dim.name] = dim.size
                else:
                    dim_names.append(dim)

            if len(dims) == 1 and variable.name in dim_names:
                dim_coord_vars.append(variable)

                compressor = None
                if variable.compressor is not None:
                    compressor = variable.compressor.make_instance()

                dim_coord = self._root.create_dataset(
                    name=variable.name,
                    dtype=variable.data_type,
                    shape=dims[0].size,
                    fill_value=None,
                    dimension_separator="/",
                    compressor=compressor,
                )

                dim_coord.attrs["_ARRAY_DIMENSIONS"] = dim_names

                long_name = variable.long_name
                if long_name is not None:
                    dim_coord.attrs["long_name"] = long_name

                var_metadata = variable.metadata
                if var_metadata is not None:
                    var_meta_dict = var_metadata.model_dump(**model_dump_kwargs)
                    dim_coord.attrs.update(var_meta_dict)

        for variable in schema.variables:
            if variable in dim_coord_vars:
                continue

            dim_names = []
            shape = []
            for dim in variable.dimensions:
                if isinstance(dim, NamedDimension):
                    name = dim.name
                    size = dim.size
                elif isinstance(dim, str):
                    name = dim
                    size = all_dims[dim]
                else:
                    raise NotImplementedError

                dim_names.append(name)
                shape.append(size)

            create_kwargs = {}
            if isinstance(variable.data_type, StructuredType):
                fields = variable.data_type.fields
                formats = [(field.name, field.format) for field in fields]
                numpy_dtype = np.dtype(formats)
                create_kwargs["fill_value"] = None
            else:
                numpy_dtype = np.dtype(variable.data_type)
                create_kwargs["fill_value"] = fill_value_map[variable.data_type]

            chunk_sizes = None
            if variable.metadata is not None:
                chunk_grid = variable.metadata.chunk_grid
                if chunk_grid is not None:
                    chunk_sizes = chunk_grid.configuration.chunk_shape
                    del variable.metadata.chunk_grid

            compressor = None
            if variable.compressor is not None:
                compressor = variable.compressor.make_instance()

            zarr_array = self._root.create_dataset(
                name=variable.name,
                dtype=numpy_dtype,
                shape=shape,
                chunks=chunk_sizes,
                dimension_separator="/",
                compressor=compressor,
                **create_kwargs,
            )
            zarr_array.attrs["_ARRAY_DIMENSIONS"] = dim_names

            if variable.coordinates is not None:
                zarr_array.attrs["coordinates"] = " ".join(variable.coordinates)

            if variable.long_name is not None:
                zarr_array.attrs["longName"] = variable.long_name

            if variable.metadata is not None:
                meta_dict = variable.metadata.model_dump(**model_dump_kwargs)
                zarr_array.attrs.update(meta_dict)

        self.consolidate_metadata()

    def set_variable_stats(self, variable: str, stats: dict[str, Any]) -> None:
        """Sets the stats attribute for a specific variable with validation."""
        self._ensure_on_disk()

        variable = self._root.get(variable, None)
        if variable is None:
            msg = f"Variable {variable} doesn't exist."
            raise KeyError(msg)

        stats_model = SummaryStatistics.model_validate(stats)
        stats_meta_model = StatisticsMetadata(stats_v1=stats_model)
        stats_dict = stats_meta_model.model_dump(**model_dump_kwargs)
        variable.attrs.update(stats_dict)

        self.consolidate_metadata()

    def set_dimension_coords(self, dimension: str, values: ArrayLike) -> None:
        """Sets the dimension coordinates for a specific dimension."""
        self._ensure_on_disk()

        dimension_var = self._root.get(dimension, None)
        if dimension_var is None:
            msg = f"Dimension {dimension_var} doesn't exist."
            raise KeyError(msg)

        dimension_var[:] = values

    def _ensure_on_disk(self) -> None:
        """Check to ensure store was created on disk."""
        if self._store is None or self._root is None:
            msg = "No dataset created. Call `make_mdio_dataset` first."
            raise ValueError(msg)

    def consolidate_metadata(self) -> None:
        """Consolidate the store metadata into .zmetadata file."""
        self._ensure_on_disk()
        zarr.consolidate_metadata(self._store)


num_proc = 8
block_size = 50_000


def calc_sample_axis(sample_interval: int, samples_per_trace: int) -> np.ndarray:
    """Calculate the sample axis from trace properties."""
    return np.arange(samples_per_trace) * sample_interval


def read_headers(file: SegyFile, start: int, stop: int) -> tuple[slice, HeaderArray]:
    """Reader headers from SEG-Y file very inefficiently."""
    slice_ = slice(start, stop)
    headers = file.trace[start:stop].header
    # headers = file.header[start:stop]
    return slice_, headers


def traces_to_zarr(  # noqa: PLR0913
    segy_file: SegyFile,
    out_path: str,
    region: dict[str, slice],
    grid_map: zarr.Array,
    dataset: xrDataset,
) -> SummaryStatistics | None:
    """Read a subset of traces and write to region of Zarr file."""
    if dataset.trace_mask.sum() == 0:
        return None

    dataset = dataset.drop_vars(["trace_mask"])
    not_null = grid_map != UINT32_MAX
    traces = segy_file.trace[grid_map[not_null].tolist()]

    # Remove extra coords if they exist
    dataset = dataset.reset_coords()
    dataset = dataset[["seismic", "headers"]]

    dataset["headers"].data[not_null] = traces.header
    dataset["headers"].data[~not_null] = 0
    dataset["seismic"].data[not_null] = traces.sample

    dataset.to_zarr(out_path, region=region, mode="r+", write_empty_chunks=False)

    histogram = CenteredBinHistogram(bin_centers=[], counts=[])
    return SummaryStatistics(
        count=traces.sample.size,
        min=traces.sample.min(),
        max=traces.sample.max(),
        sum=traces.sample.sum(),
        sum_squares=(traces.sample**2).sum(),
        histogram=histogram,
    )


# TODO(Altay): Simplify this A LOT  # noqa: TD003
def segy_to_mdio(  # noqa: PLR0913, PLR0915
    in_path: str,
    out_path: str,
    name: str,
    dataset_type: MDIOSchemaType,  # noqa: A002
    create_coords: bool = False,
    z_units: dict[str, str] | None = None,
    segy_kwargs: dict[str, Any] | None = None,
) -> None:
    """Ingest SEG-Y file to MDIO."""
    segy_file = SegyFile(in_path, **segy_kwargs)
    template = SCHEMA_TEMPLATE_MAP[dataset_type]

    domain = template._trace_domain
    dim_names = template._dim_names

    index_keys = dim_names[:-1]

    # ms to s (in this case millimeter to m)
    sample_interval = segy_file.binary_header["sample_interval"].squeeze() // 1000
    samples_per_trace = segy_file.binary_header["samples_per_trace"].squeeze()

    samples = calc_sample_axis(sample_interval.item(), samples_per_trace.item())

    no_blocks = int(np.floor(segy_file.num_traces / block_size)) + 1
    header_fields = segy_file.spec.trace.header.fields
    header_dtype = np.dtype([(field.name, field.format) for field in header_fields])
    headers = np.empty(dtype=header_dtype, shape=segy_file.num_traces)
    unique_dim_coords = {key: set() for key in index_keys}
    unique_dim_coords[domain] = set(samples)
    with ProcessPoolExecutor(num_proc) as executor:
        futures = []
        for idx in range(no_blocks):
            start = idx * block_size
            stop = min(segy_file.num_traces, start + block_size)
            future = executor.submit(read_headers, segy_file, start, stop)
            futures.append(future)

        iterable = tqdm(
            as_completed(futures),
            total=no_blocks,
            unit="block",
            desc="Scanning headers",
        )

        for future in iterable:
            slice_, header_subset = future.result()
            headers[slice_] = header_subset

    for key in index_keys:
        unique_items = np.unique(headers[key])
        unique_dim_coords[key].update(unique_items)

    unique_dim_coords = {k: sorted(v) for k, v in unique_dim_coords.items()}
    shape = tuple(len(v) for v in unique_dim_coords.values())

    # Get header fields dict -> {name: dtype}
    header_fields = {field.name: field.format for field in header_fields}

    factory = MDIOFactory(out_path, dataset_type)
    factory.create_schema(
        name=name,
        shape=shape,
        header_fields=header_fields,
        z_units=z_units,
        create_coords=create_coords,
    )

    out_schema_path = str(out_path).replace(".mdio", "_schema.json")

    # TODO(Altay): This could be a cloud link; need to implement other `open` funcs.   # noqa: TD003
    out_dir = Path(out_schema_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_schema_path, mode="w") as fp:  # noqa: PTH123
        fp.write(factory.schema_json)

    factory.make_mdio_dataset()
    factory.set_dimension_coords(domain, samples)
    for key in index_keys:
        factory.set_dimension_coords(key, unique_dim_coords[key])
        factory.set_dimension_coords(key, unique_dim_coords[key])

    # build live mask and trace indices
    live_indices = ()
    for dim_name in index_keys:
        dim_coords = unique_dim_coords[dim_name]
        dim_hdr = headers[dim_name]
        live_indices += (np.searchsorted(dim_coords, dim_hdr),)

    # We set dead traces to uint32 max. Should be far away from actual trace counts.
    grid_map = zarr.full(shape[:-1], dtype="uint32", fill_value=UINT32_MAX)
    grid_map.vindex[live_indices] = np.arange(live_indices[0].size)

    trace_mask = zarr.zeros(shape[:-1], dtype="bool")
    trace_mask.vindex[live_indices] = 1

    ds = xr.open_zarr(out_path, chunks=None)
    ds.trace_mask.data[:] = trace_mask

    if create_coords:
        coord_scaler = headers[0]["scalar_apply_coords"].item()
        if coord_scaler < 0:
            # negative means division
            coord_scaler = 1 / abs(coord_scaler)
        elif coord_scaler == 0:
            # edge case where its 0
            coord_scaler = 1

        for coord in template._coords:
            if coord in {"gun"}:
                _, shot_start_idx = np.unique(headers["shot_point"], return_index=True)
                shot_mask = trace_mask[:, 0, 0]
                ds[coord].data[shot_mask] = headers[shot_start_idx][coord]
            elif coord in {"shot-x", "shot-y"}:
                _, shot_start_idx = np.unique(headers["shot_point"], return_index=True)
                shot_mask = trace_mask[:, 0, 0]
                ds[coord].data[shot_mask] = headers[shot_start_idx][coord]
                ds[coord].data[:] *= coord_scaler
            else:
                ds[coord].data[trace_mask] = headers[coord] * coord_scaler
                ds[coord].data[trace_mask] = headers[coord] * coord_scaler

    ds[["trace_mask"]].to_zarr(out_path, mode="r+", write_empty_chunks=False)

    def var() -> None:
        pass

    var.shape = ds.seismic.shape
    var.chunks = tuple(ds.seismic.encoding["preferred_chunks"].values())
    var.chunks = segy_export_rechunker(
        var.chunks, var.shape, np.dtype("float32"), "128M"
    )

    chunk_iter = ChunkIterator(var, dim_names, False)

    histogram = CenteredBinHistogram(bin_centers=[], counts=[])
    final_stats = SummaryStatistics(
        count=0, min=0, max=0, sum=0, sum_squares=0, histogram=histogram
    )

    def update_stats(partial_stats: SummaryStatistics) -> None:
        final_stats.count += partial_stats.count
        final_stats.min = min(final_stats.min, partial_stats.min)
        final_stats.max = min(final_stats.max, partial_stats.max)
        final_stats.sum += partial_stats.sum
        final_stats.sum_squares += partial_stats.sum_squares

    with ProcessPoolExecutor(num_proc) as executor:
        futures = []
        common_args = (segy_file, out_path)
        for region in chunk_iter:
            index_slices = tuple(region[key] for key in index_keys)
            subset_args = (
                region,
                grid_map[index_slices],
                ds.isel(region),
            )
            future = executor.submit(traces_to_zarr, *common_args, *subset_args)
            futures.append(future)

        iterable = tqdm(
            as_completed(futures),
            total=len(chunk_iter),
            unit="block",
            desc="Ingesting traces",
        )

        for future in iterable:
            result = future.result()
            if result is not None:
                update_stats(result)

    factory.set_variable_stats("seismic", final_stats.model_dump(by_alias=True))
