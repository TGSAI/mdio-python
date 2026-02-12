"""Seismic3DReceiverGathersTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas import compressors
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import CoordinateMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate


class Seismic3DReceiverGathersTemplate(AbstractDatasetTemplate):
    """Seismic receiver gather pre-stack 3D Dataset template.

    A template for surveys with fixed receiver positions (OBN, OBC, land fixed-spread)
    where data is organized by receiver position for receiver-side processing.
    Shots are organized by shot lines with a calculated shot index.

    This template is time-domain only as receiver gathers are typically used
    for early-stage processing before depth conversion.

    Dimensions:
        - receiver: Index of the receiver node/station
        - shot_line: Shot line or swath identifier
        - shot_index: Sequential index of shots within the line (calculated, 0-N)
        - time: Sample dimension

    This organization is optimal for:
        - Receiver-side wavefield separation (up/down)
        - Receiver-consistent deconvolution
        - Multi-component processing
        - Mirror imaging from OBN
    """

    def __init__(self) -> None:
        super().__init__(data_domain="time")

        self._dim_names = ("receiver", "shot_line", "shot_index", "time")
        self._calculated_dims = ("shot_index",)
        self._physical_coord_names = (
            "receiver_x",
            "receiver_y",
            "source_coord_x",
            "source_coord_y",
        )
        self._logical_coord_names = ("shot_point",)
        self._var_chunk_shape = (1, 1, 512, 4096)

    @property
    def _name(self) -> str:
        return "ReceiverGathers3D"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "receiver_gathers"}

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        # Note: shot_index is calculated (0-N), so we don't add a coordinate for it
        self._builder.add_coordinate(
            "receiver",
            dimensions=("receiver",),
            data_type=ScalarType.UINT32,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("receiver")),
        )
        self._builder.add_coordinate(
            "shot_line",
            dimensions=("shot_line",),
            data_type=ScalarType.UINT32,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("shot_line")),
        )
        self._builder.add_coordinate(
            self.trace_domain,
            dimensions=(self.trace_domain,),
            data_type=ScalarType.INT32,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key(self.trace_domain)),
        )

        # Add non-dimension coordinates
        compressor = compressors.Blosc(cname=compressors.BloscCname.zstd)

        # Receiver coordinates (fixed per receiver)
        self._builder.add_coordinate(
            "receiver_x",
            dimensions=("receiver",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("receiver_x")),
        )
        self._builder.add_coordinate(
            "receiver_y",
            dimensions=("receiver",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("receiver_y")),
        )

        # Shot point coordinate (actual shot point numbers, varies by shot_line and shot_index)
        self._builder.add_coordinate(
            "shot_point",
            dimensions=("shot_line", "shot_index"),
            data_type=ScalarType.UINT32,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("shot_point")),
        )

        # Source coordinates (vary by shot_line and shot_index)
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=("shot_line", "shot_index"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_x")),
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=("shot_line", "shot_index"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=CoordinateMetadata(units_v1=self.get_unit_by_key("source_coord_y")),
        )
