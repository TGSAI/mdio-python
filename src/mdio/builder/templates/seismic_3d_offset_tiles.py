"""Seismic3DOffsetTilesTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas import compressors
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import CoordinateSpec
from mdio.builder.templates.types import DimCoordinateTypes
from mdio.builder.templates.types import SeismicDataDomain


class Seismic3DOffsetTilesTemplate(AbstractDatasetTemplate):
    """Seismic 3D template for rectangular offset tile gathers."""

    def __init__(self, data_domain: SeismicDataDomain = "time"):
        super().__init__(data_domain=data_domain)

        self._dim_names = (
            "inline",
            "crossline",
            "inline_offset_tile",
            "crossline_offset_tile",
            self._data_domain,
        )
        self._physical_coord_names = ("cdp_x", "cdp_y")
        self._logical_coord_names = ()
        self._var_chunk_shape = (4, 4, 6, 6, 4096)

    @property
    def _name(self) -> str:
        return f"OffsetTiles3D{self._data_domain.capitalize()}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "offset_tiles"}

    def declare_coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        """Declare inline/crossline-indexed X/Y coordinates for the 3D offset tiles template."""
        return (
            CoordinateSpec(name="cdp_x", dimensions=("inline", "crossline"), dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="cdp_y", dimensions=("inline", "crossline"), dtype=ScalarType.FLOAT64),
        )

    def declare_dim_coordinate_types(self) -> DimCoordinateTypes:
        """Declare the data types for each dimension coordinate in this template."""
        return {
            "inline": ScalarType.INT32,
            "crossline": ScalarType.INT32,
            "inline_offset_tile": ScalarType.INT16,
            "crossline_offset_tile": ScalarType.INT16,
            self._data_domain: ScalarType.INT32,
        }

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        self._builder.add_coordinate(
            "inline",
            dimensions=("inline",),
            data_type=self._dim_dtype("inline"),
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("inline")),
        )
        self._builder.add_coordinate(
            "crossline",
            dimensions=("crossline",),
            data_type=self._dim_dtype("crossline"),
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("crossline")),
        )
        self._builder.add_coordinate(
            "inline_offset_tile",
            dimensions=("inline_offset_tile",),
            data_type=self._dim_dtype("inline_offset_tile"),
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("inline_offset_tile")),
        )
        self._builder.add_coordinate(
            "crossline_offset_tile",
            dimensions=("crossline_offset_tile",),
            data_type=self._dim_dtype("crossline_offset_tile"),
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("crossline_offset_tile")),
        )
        self._builder.add_coordinate(
            self.trace_domain,
            dimensions=(self.trace_domain,),
            data_type=self._dim_dtype(self.trace_domain),
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(self.trace_domain)),
        )

        # Add non-dimension coordinates
        compressor = compressors.Blosc(cname=compressors.BloscCname.zstd)

        # CDP coordinates (vary by inline, crossline)
        self._builder.add_coordinate(
            "cdp_x",
            dimensions=("inline", "crossline"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("cdp_x")),
        )
        self._builder.add_coordinate(
            "cdp_y",
            dimensions=("inline", "crossline"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("cdp_y")),
        )
