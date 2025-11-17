"""Seismic2DCDPGathersTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import CdpGatherDomain
from mdio.builder.templates.types import SeismicDataDomain


class Seismic2DCdpGathersTemplate(AbstractDatasetTemplate):
    """Seismic CDP pre-stack 2D time or depth Dataset template."""

    def __init__(self, data_domain: SeismicDataDomain, gather_domain: CdpGatherDomain):
        super().__init__(data_domain=data_domain)
        self._gather_domain = gather_domain.lower()

        if self._gather_domain not in ["offset", "angle"]:
            msg = "gather_type must be 'offset' or 'angle'"
            raise ValueError(msg)

        self._dim_names = ("cdp", self._gather_domain, self._data_domain)
        self._physical_coord_names = ("cdp_x", "cdp_y")
        self._var_chunk_shape = (16, 64, 1024)

    @property
    def _name(self) -> str:
        gather_domain_suffix = self._gather_domain.capitalize()
        data_domain_suffix = self._data_domain.capitalize()
        return f"Cdp{gather_domain_suffix}Gathers2D{data_domain_suffix}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "2D", "gatherType": "cdp"}

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        self._builder.add_coordinate(
            "cdp",
            dimensions=("cdp",),
            data_type=ScalarType.INT32,
        )
        self._builder.add_coordinate(
            self._gather_domain,
            dimensions=(self._gather_domain,),
            data_type=ScalarType.INT32,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(self._gather_domain)),
        )
        self._builder.add_coordinate(
            self.trace_domain,
            dimensions=(self.trace_domain,),
            data_type=ScalarType.INT32,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(self.trace_domain)),
        )

        # Add non-dimension coordinates
        compressor = Blosc(cname=BloscCname.zstd)
        self._builder.add_coordinate(
            "cdp_x",
            dimensions=("cdp",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("cdp_x")),
        )
        self._builder.add_coordinate(
            "cdp_y",
            dimensions=("cdp",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("cdp_y")),
        )
