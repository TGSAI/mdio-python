"""Seismic3DPostStackTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.builder.templates.abstract_dataset_template import SeismicDataDomain


class Seismic3DPostStackTemplate(AbstractDatasetTemplate):
    """Seismic post-stack 3D time or depth Dataset template."""

    def __init__(self, data_domain: SeismicDataDomain):
        super().__init__(data_domain=data_domain)

        self._coord_dim_names = ("inline", "crossline")
        self._dim_names = (*self._coord_dim_names, self._data_domain)
        self._coord_names = ("cdp_x", "cdp_y")
        self._var_chunk_shape = (128, 128, 128)

    @property
    def _name(self) -> str:
        domain_suffix = self._data_domain.capitalize()
        return f"PostStack3D{domain_suffix}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "stacked"}
