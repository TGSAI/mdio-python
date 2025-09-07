"""Seismic2DPostStackTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate


class Seismic2DPostStackTemplate(AbstractDatasetTemplate):
    """Seismic post-stack 2D time or depth Dataset template."""

    def __init__(self, domain: str):
        super().__init__(domain=domain)

        self._coord_dim_names = ("cdp",)
        self._dim_names = (*self._coord_dim_names, self._trace_domain)
        self._coord_names = ("cdp_x", "cdp_y")
        self._var_chunk_shape = (1024, 1024)

    @property
    def _name(self) -> str:
        return f"PostStack2D{self._trace_domain.capitalize()}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyDimensionality": "2D", "ensembleType": "line", "processingStage": "post-stack"}
