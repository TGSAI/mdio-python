"""SeismicPreStackTemplate MDIO v1 dataset templates."""

from mdio.schemas.dtype import ScalarType
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate


class SeismicPreStackTemplate(AbstractDatasetTemplate):
    """Seismic pre-stack time Dataset template.

    This should be used for both 2D and 3D datasets. Common-shot or common-channel datasets

    Args:
        domain: The domain of the dataset.
    """

    def __init__(self, domain: str = "time"):
        super().__init__(domain=domain)

        self._coord_dim_names = [
            "shot_line",
            "gun",
            "shot_point",
            "cable",
            "channel",
        ]  # Custom coordinates for shot gathers
        self._dim_names = [*self._coord_dim_names, self._trace_domain]
        self._coord_names = ["source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y"]
        self._var_chunk_shape = [1, 1, 16, 1, 32, -1]

    @property
    def _name(self) -> str:
        return f"PreStackGathers3D{self._trace_domain.capitalize()}"

    def _load_dataset_attributes(self) -> UserAttributes:
        return UserAttributes(
            attributes={
                "surveyDimensionality": "3D",
                "ensembleType": "shot_point",
                "processingStage": "pre-stack",
            }
        )

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        for name in self._dim_names:
            self._builder.add_coordinate(
                name,
                dimensions=[name],
                data_type=ScalarType.INT32,
                metadata_info=None,
            )

        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=["shot_line", "gun", "shot_point", "cable", "channel"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit],
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=["shot_line", "gun", "shot_point", "cable", "channel"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit],
        )
        self._builder.add_coordinate(
            "group_coord_x",
            dimensions=["shot_line", "gun", "shot_point", "cable", "channel"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit],
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=["shot_line", "gun", "shot_point", "cable", "channel"],
            data_type=ScalarType.FLOAT64,
            metadata_info=[self._horizontal_coord_unit],
        )
