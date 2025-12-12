"""Seismic2DCDPGathersTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import CdpGatherDomain
from mdio.builder.templates.types import SeismicDataDomain
from mdio.core.utils_write import MAX_COORDINATES_BYTES
from mdio.core.utils_write import get_constrained_chunksize


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
            metadata=VariableMetadata(units_v1=self.get_unit_by_key(self._gather_domain)),
        )
        self._builder.add_coordinate(
            self.trace_domain,
            dimensions=(self.trace_domain,),
            data_type=ScalarType.INT32,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key(self.trace_domain)),
        )

        # Add non-dimension coordinates with computed chunk sizes
        # For 2D CDP, coordinates are only over cdp dimension
        coord_spatial_shape = (self._dim_sizes[0],)  # cdp only
        coord_chunk_shape = get_constrained_chunksize(
            coord_spatial_shape,
            ScalarType.FLOAT64,
            MAX_COORDINATES_BYTES,
        )
        chunk_grid = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=coord_chunk_shape))

        compressor = Blosc(cname=BloscCname.zstd)
        self._builder.add_coordinate(
            "cdp_x",
            dimensions=("cdp",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("cdp_x"), chunk_grid=chunk_grid),
        )
        self._builder.add_coordinate(
            "cdp_y",
            dimensions=("cdp",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("cdp_y"), chunk_grid=chunk_grid),
        )
