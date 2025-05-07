"""MDIO factories for seismic data."""

# TODO(BrianMichell): Add implementations for other canonical datasets.

from __future__ import annotations

from enum import Enum
from enum import auto
from typing import Any

from mdio.core.v1.builder import MDIODatasetBuilder
from mdio.schema.compressors import Blosc
from mdio.schema.dtype import ScalarType
from mdio.schema.dtype import StructuredType
from mdio.schema.v1.dataset import Dataset


class MDIOSchemaType(Enum):
    """MDIO templates for specific data types."""

    SEISMIC_3D_POST_STACK_GENERIC = auto()
    SEISMIC_3D_POST_STACK_TIME = auto()
    SEISMIC_3D_POST_STACK_DEPTH = auto()
    SEISMIC_3D_PRE_STACK_CDP_TIME = auto()
    SEISMIC_3D_PRE_STACK_CDP_DEPTH = auto()


class Seismic3DPostStackGeneric:
    """Generic 3D seismic post stack dataset."""

    def __init__(self):
        """Initialize generic post stack dataset."""
        self._dim_names = ["inline", "crossline", "sample"]
        self._chunks = [128, 128, 128]  # 8 mb
        self._coords = {
            "cdp-x": ("float32", {"unitsV1": {"length": "m"}}, self._dim_names[:-1]),
            "cdp-y": ("float32", {"unitsV1": {"length": "m"}}, self._dim_names[:-1]),
        }

    def create(
        self,
        name: str,
        shape: list[int],
        header_fields: dict[str, str],
        create_coords: bool = False,
        sample_format: str | None = None,
        chunks: list[int] | None = None,
        sample_units: dict[str, str] | None = None,
        z_units: dict[str, str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Dataset:
        """Create a generic seismic dataset schema.

        Args:
            name: Name of the dataset
            shape: Shape of the dataset
            header_fields: Header fields to include as a dict of field_name: dtype
            create_coords: Whether to create coordinates
            sample_format: Format of the samples
            chunks: Chunk sizes
            sample_units: Units for samples
            z_units: Units for z-axis
            attributes: Additional attributes to include in the dataset metadata

        Returns:
            Dataset: The created dataset
        """
        chunks = chunks or self._chunks
        sample_format = sample_format or "float32"

        builder = MDIODatasetBuilder(
            name=name,
            attributes=attributes,
        )

        # Add dimensions
        for dim_name, dim_size in zip(self._dim_names, shape, strict=True):
            builder.add_dimension(
                name=dim_name,
                size=dim_size,
                data_type=ScalarType.UINT32,
                metadata=z_units if dim_name == "sample" else None,
            )

        # Add coordinates if requested
        if create_coords:
            for coord_name, (format_, unit, coord_dims) in self._coords.items():
                builder.add_coordinate(
                    name=coord_name,
                    data_type=ScalarType(format_),
                    dimensions=coord_dims,
                    metadata=unit,
                )

        # Add seismic variable
        builder.add_variable(
            name="seismic",
            data_type=ScalarType(sample_format),
            dimensions=self._dim_names,
            compressor=Blosc(name="blosc", algorithm="zstd"),
            metadata=sample_units,
        )

        # Add header variable with structured dtype
        header_dtype = StructuredType(
            fields=[
                {"name": field_name, "format": field_type}
                for field_name, field_type in header_fields.items()
            ]
        )
        builder.add_variable(
            name="headers",
            data_type=header_dtype,
            dimensions=self._dim_names[:-1],
            compressor=Blosc(name="blosc"),
        )

        # Add trace mask
        builder.add_variable(
            name="trace_mask",
            data_type=ScalarType.BOOL,
            dimensions=self._dim_names[:-1],
            compressor=Blosc(name="blosc"),
        )

        return builder.build()


class Seismic3DPostStack(Seismic3DPostStackGeneric):
    """3D seismic post stack dataset with domain-specific attributes."""

    def __init__(self, domain: str):
        """Initialize post stack dataset.

        Args:
            domain: Domain of the dataset (time/depth)
        """
        super().__init__()
        self._dim_names = ["inline", "crossline", domain]

    def create(
        self,
        name: str,
        shape: list[int],
        header_fields: dict[str, str],
        create_coords: bool = False,
        sample_format: str | None = None,
        chunks: list[int] | None = None,
        sample_units: dict[str, str] | None = None,
        z_units: dict[str, str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Dataset:
        """Create a seismic dataset schema with domain-specific attributes."""
        # Add seismic-specific attributes
        seismic_attrs = {
            "surveyDimensionality": "3D",
            "ensembleType": "line",
            "processingStage": "post-stack",
        }
        if attributes:
            seismic_attrs.update(attributes)

        return super().create(
            name=name,
            shape=shape,
            header_fields=header_fields,
            create_coords=create_coords,
            sample_format=sample_format,
            chunks=chunks,
            sample_units=sample_units,
            z_units=z_units,
            attributes=seismic_attrs,
        )


class Seismic3DPreStack(Seismic3DPostStackGeneric):
    """3D seismic pre stack dataset."""

    def __init__(self, domain: str):
        """Initialize pre stack dataset.

        Args:
            domain: Domain of the dataset (time/depth)
        """
        super().__init__()
        self._dim_names = ["inline", "crossline", "offset", domain]
        self._chunks = [1, 1, 512, 4096]  # 8 mb
        self._coords = {
            "cdp-x": ("float32", {"length": "m"}, self._dim_names[:-2]),
            "cdp-y": ("float32", {"length": "m"}, self._dim_names[:-2]),
        }

    def create(
        self,
        name: str,
        shape: list[int],
        header_fields: dict[str, str],
        create_coords: bool = False,
        sample_format: str | None = None,
        chunks: list[int] | None = None,
        sample_units: dict[str, str] | None = None,
        z_units: dict[str, str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Dataset:
        """Create a seismic dataset schema with pre-stack attributes."""
        # Add seismic-specific attributes
        seismic_attrs = {
            "surveyDimensionality": "3D",
            "ensembleType": "cdp",
            "processingStage": "pre-stack",
        }
        if attributes:
            seismic_attrs.update(attributes)

        return super().create(
            name=name,
            shape=shape,
            header_fields=header_fields,
            create_coords=create_coords,
            sample_format=sample_format,
            chunks=chunks,
            sample_units=sample_units,
            z_units=z_units,
            attributes=seismic_attrs,
        )


SCHEMA_TEMPLATE_MAP = {
    MDIOSchemaType.SEISMIC_3D_POST_STACK_GENERIC: Seismic3DPostStackGeneric(),
    MDIOSchemaType.SEISMIC_3D_POST_STACK_TIME: Seismic3DPostStack("time"),
    MDIOSchemaType.SEISMIC_3D_POST_STACK_DEPTH: Seismic3DPostStack("depth"),
    MDIOSchemaType.SEISMIC_3D_PRE_STACK_CDP_TIME: Seismic3DPreStack("time"),
    MDIOSchemaType.SEISMIC_3D_PRE_STACK_CDP_DEPTH: Seismic3DPreStack("depth"),
}
