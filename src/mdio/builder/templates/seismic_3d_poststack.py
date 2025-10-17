"""Seismic3DPostStackTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain


class Seismic3DPostStackTemplate(AbstractDatasetTemplate):
    """Seismic post-stack 3D time or depth Dataset template."""

    def __init__(self, data_domain: SeismicDataDomain):
        super().__init__(data_domain=data_domain)

        self._dim_names = ("inline", "crossline", self._data_domain)
        self._physical_coord_names = ("cdp_x", "cdp_y")
        self._var_chunk_shape = (128, 128, 128)

    @property
    def _name(self) -> str:
        domain_suffix = self._data_domain.capitalize()
        return f"PostStack3D{domain_suffix}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "stacked"}
