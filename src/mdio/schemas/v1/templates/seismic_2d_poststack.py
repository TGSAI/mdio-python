"""Seismic2DPostStackTemplate MDIO v1 dataset templates."""

from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate


class Seismic2DPostStackTemplate(AbstractDatasetTemplate):
    """Seismic post-stack 2D time or depth Dataset template."""

    def __init__(self, domain: str):
        super().__init__(domain=domain)

        self._coord_dim_names = ["cdp"]
        self._dim_names = [*self._coord_dim_names, self._trace_domain]
        self._coord_names = ["cdp-x", "cdp-y"]
        self._var_name = "StackedAmplitude"
        self._var_chunk_shape = [1024, 1024]

    def _get_name(self) -> str:
        return f"PostStack2D{self._trace_domain.capitalize()}"

    def _load_dataset_attributes(self) -> UserAttributes:
        return UserAttributes(
            attributes={
                "surveyDimensionality": "2D",
                "ensembleType": "line",
                "processingStage": "post-stack",
            }
        )
