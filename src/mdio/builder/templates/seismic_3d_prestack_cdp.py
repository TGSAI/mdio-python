"""Seismic3DPreStackCDPTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import CdpGatherDomain
from mdio.builder.templates.types import SeismicDataDomain


class Seismic3DPreStackCDPTemplate(AbstractDatasetTemplate):
    """Seismic CDP pre-stack 3D gathers Dataset template."""

    def __init__(self, data_domain: SeismicDataDomain, gather_domain: CdpGatherDomain):
        super().__init__(data_domain=data_domain)
        self._gather_domain = gather_domain.lower()

        if self._gather_domain not in ["offset", "angle"]:
            msg = "gather_type must be 'offset' or 'angle'"
            raise ValueError(msg)

        self._dim_names = ("inline", "crossline", self._gather_domain, self._data_domain)
        self._physical_coord_names = ("cdp_x", "cdp_y")
        self._var_chunk_shape = (8, 8, 32, 512)

    @property
    def _name(self) -> str:
        gather_domain_suffix = self._gather_domain.capitalize()
        data_domain_suffix = self._data_domain.capitalize()
        return f"PreStackCdp{gather_domain_suffix}Gathers3D{data_domain_suffix}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "cdp"}
