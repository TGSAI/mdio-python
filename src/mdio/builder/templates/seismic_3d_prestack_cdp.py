"""Seismic3DPreStackCDPTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.templates.abstract_dataset_template import AbstractDatasetTemplate


class Seismic3DPreStackCDPTemplate(AbstractDatasetTemplate):
    """Seismic CDP pre-stack 3D time or depth Dataset template."""

    def __init__(self, domain: str):
        super().__init__(domain=domain)

        self._coord_dim_names = ("inline", "crossline", "offset")
        self._dim_names = (*self._coord_dim_names, self._trace_domain)
        self._coord_names = ("cdp_x", "cdp_y")
        self._var_chunk_shape = (1, 1, 512, 4096)

    @property
    def _name(self) -> str:
        return f"PreStackCdpGathers3D{self._trace_domain.capitalize()}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyDimensionality": "3D", "ensembleType": "cdp", "processingStage": "pre-stack"}
