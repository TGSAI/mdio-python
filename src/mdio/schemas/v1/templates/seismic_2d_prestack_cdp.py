"""Seismic2DPreStackCDPTemplate MDIO v1 dataset templates."""

from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate


class Seismic2DPreStackCDPTemplate(AbstractDatasetTemplate):
    """Seismic CDP pre-stack 2D time or depth Dataset template."""

    def __init__(self, domain: str):
        super().__init__(domain=domain)

        self._coord_dim_names = ["cdp", "offset"]
        self._dim_names = [*self._coord_dim_names, self._trace_domain]
        self._coord_names = ["cdp_x", "cdp_y"]
        self._var_chunk_shape = [1, 512, 4096]

    @property
    def _name(self) -> str:
        return f"PreStackCdpGathers2D{self._trace_domain.capitalize()}"

    def _load_dataset_attributes(self) -> UserAttributes:
        return UserAttributes(
            attributes={
                "surveyDimensionality": "2D",
                "ensembleType": "cdp",
                "processingStage": "pre-stack",
            }
        )
