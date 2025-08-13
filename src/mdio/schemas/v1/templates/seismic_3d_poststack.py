"""Seismic3DPostStackTemplate MDIO v1 dataset templates."""

from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate


class Seismic3DPostStackTemplate(AbstractDatasetTemplate):
    """Seismic post-stack 3D time or depth Dataset template."""

    def __init__(self, domain: str):
        super().__init__(domain=domain)
        # Template attributes to be overridden by subclasses
        self._coord_dim_names = ["inline", "crossline"]
        self._dim_names = [*self._coord_dim_names, self._trace_domain]
        self._coord_names = ["cdp_x", "cdp_y"]
        self._var_chunk_shape = [128, 128, 128]
        self._grid_chunk_shape = [1024, 1024]

    @property
    def _name(self) -> str:
        return f"PostStack3D{self._trace_domain.capitalize()}"

    def _load_dataset_attributes(self) -> UserAttributes:
        return UserAttributes(
            attributes={
                "surveyDimensionality": "3D",
                "ensembleType": "line",
                "processingStage": "post-stack",
            }
        )
