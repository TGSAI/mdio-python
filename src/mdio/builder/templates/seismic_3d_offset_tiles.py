"""Seismic3DOffsetTilesTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas import compressors
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain


class Seismic3DOffsetTilesTemplate(AbstractDatasetTemplate):
    """Seismic Offset Tiles pre-stack 3D Dataset template.

    A 5D template for wide-azimuth seismic data organized by offset vector tiles.
    Data is binned by CDP location (inline, crossline) with offset vector
    decomposition into inline and crossline offset tile components.

    Dimensions:
        - inline: Inline bin position
        - crossline: Crossline bin position
        - inline_offset_tile: Inline component of the offset vector tile
        - crossline_offset_tile: Crossline component of the offset vector tile
        - time/depth: Sample dimension

    This organization is optimal for:
        - Wide-azimuth data preservation
        - 5D interpolation and regularization
        - Azimuthal analysis and processing
        - Offset vector filtering

    The offset vector tiles partition the offset-azimuth space into a regular grid,
    preserving both offset magnitude and azimuth information in a format suitable
    for modern wide-azimuth processing workflows.

    Args:
        data_domain: The domain of the dataset ('time' or 'depth').
    """

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

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        self._builder.add_coordinate(
            "inline",
            dimensions=("inline",),
            data_type=ScalarType.INT32,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("inline")),
        )
        self._builder.add_coordinate(
            "crossline",
            dimensions=("crossline",),
            data_type=ScalarType.INT32,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("crossline")),
        )
        self._builder.add_coordinate(
            "inline_offset_tile",
            dimensions=("inline_offset_tile",),
            data_type=ScalarType.INT16,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("inline_offset_tile")),
        )
        self._builder.add_coordinate(
            "crossline_offset_tile",
            dimensions=("crossline_offset_tile",),
            data_type=ScalarType.INT16,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("crossline_offset_tile")),
        )
        self._builder.add_coordinate(
            self.trace_domain,
            dimensions=(self.trace_domain,),
            data_type=ScalarType.INT32,
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
