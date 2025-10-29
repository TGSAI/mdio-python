"""Seismic3DCocaTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas import compressors
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain


class Seismic3DCocaGathersTemplate(AbstractDatasetTemplate):
    """Seismic CoCA (common offset, common azimuth) pre-stack 3D Dataset template."""

    def __init__(self, data_domain: SeismicDataDomain):
        super().__init__(data_domain=data_domain)

        self._dim_names = ("inline", "crossline", "offset", "azimuth", self._data_domain)
        self._physical_coord_names = ("cdp_x", "cdp_y")
        self._var_chunk_shape = (8, 8, 32, 1, 1024)

    @property
    def _name(self) -> str:
        return f"CocaGathers3D{self._data_domain.capitalize()}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "common_offset_common_azimuth"}

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        self._builder.add_coordinate(
            "inline",
            dimensions=("inline",),
            data_type=ScalarType.INT32,
        )
        self._builder.add_coordinate(
            "crossline",
            dimensions=("crossline",),
            data_type=ScalarType.INT32,
        )
        self._builder.add_coordinate(
            "offset",
            dimensions=("offset",),
            data_type=ScalarType.INT32,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("offset")),  # same unit as X/Y
        )
        self._builder.add_coordinate(
            "azimuth",
            dimensions=("azimuth",),
            data_type=ScalarType.FLOAT32,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("azimuth")),
        )
        self._builder.add_coordinate(
            self.trace_domain,
            dimensions=(self.trace_domain,),
            data_type=ScalarType.INT32,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(self.trace_domain)),
        )

        # Add non-dimension coordinates
        compressor = compressors.Blosc(cname=compressors.BloscCname.zstd)
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
