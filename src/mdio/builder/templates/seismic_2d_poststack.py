"""Seismic2DPostStackTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.builder.templates.abstract_dataset_template import SeismicDataDomain


class Seismic2DPostStackTemplate(AbstractDatasetTemplate):
    """Seismic post-stack 2D time or depth Dataset template."""

    def __init__(self, data_domain: SeismicDataDomain):
        super().__init__(data_domain=data_domain)

        self._coord_dim_names = ("cdp",)
        self._dim_names = (*self._coord_dim_names, self._data_domain)
        self._coord_names = ("cdp_x", "cdp_y")
        self._var_chunk_shape = (1024, 1024)

    @property
    def _name(self) -> str:
        return f"PostStack2D{self._data_domain.capitalize()}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "2D", "gatherType": "stacked"}
