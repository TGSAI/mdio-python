"""Template method pattern implementation for MDIO v1 dataset template."""

import copy
from abc import ABC
from abc import abstractmethod

from mdio.schemas import compressors
from mdio.schemas.chunk_grid import RegularChunkGrid
from mdio.schemas.chunk_grid import RegularChunkShape
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.metadata import ChunkGridMetadata
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.dataset_builder import MDIODatasetBuilder
from mdio.schemas.v1.units import AllUnits


class AbstractDatasetTemplate(ABC):
    """Abstract base class that defines the template method for Dataset building factory.

    The template method defines the skeleton of the data processing algorithm,
    while allowing subclasses to override specific steps.
    """

    def __init__(self, domain: str = "") -> None:
        # Template attributes to be overridden by subclasses
        # Domain of the seismic data, e.g. "time" or "depth"
        self._trace_domain = domain.lower()
        # Names of all coordinate dimensions in the dataset
        # e.g. ["cdp"] for 2D post-stack depth
        # e.g. ["inline", "crossline"] for 3D post-stack
        # e.g. ["inline", "crossline"] for 3D pre-stack CDP gathers
        # Note: For pre-stack Shot gathers, the coordinates are defined differently
        #       and are not directly tied to _coord_dim_names.
        self._coord_dim_names = []
        # *ORDERED* list of names of all dimensions in the dataset
        # e.g. ["cdp", "depth"] for 2D post-stack depth
        # e.g. ["inline", "crossline", "depth"] for 3D post-stack depth
        # e.g. ["inline", "crossline", "offset", "depth"] for 3D pre-stack depth CPD gathers
        # e.g. ["shot_point", "cable", "channel", "time"] for 3D pre-stack time Shot gathers
        self._dim_names = []
        # Names of all coordinates in the dataset
        # e.g. ["cdp_x", "cdp_y"] for 2D post-stack depth
        # e.g. ["cdp_x", "cdp_y"] for 3D post-stack depth
        # e.g. ["cdp_x", "cdp_y"] for 3D pre-stack CPD depth
        # e.g. ["gun", "shot-x", "shot-y", "receiver-x", "receiver-y"] for 3D pre-stack
        #      time Shot gathers
        self._coord_names = []
        # Chunk shape for the variable in the dataset
        # e.g. [1024, 1024] for 2D post-stack depth
        # e.g. [128, 128, 128] for 3D post-stack depth
        # e.g. [1, 1, 512, 4096] for 3D pre-stack CPD depth
        # e.g. [1, 1, 512, 4096] for 3D pre-stack time Shot gathers
        self._var_chunk_shape = []

        # Variables instantiated when build_dataset() is called
        self._builder: MDIODatasetBuilder = None
        # Sizes of the dimensions in the dataset, to be set when build_dataset() is called
        self._dim_sizes = []
        # Horizontal units for the coordinates (e.g, "m", "ft"), to be set when
        # build_dataset() is called
        self._horizontal_coord_unit = None

    def build_dataset(
        self,
        name: str,
        sizes: list[int],
        horizontal_coord_unit: AllUnits,
        headers: StructuredType = None,
    ) -> Dataset:
        """Template method that builds the dataset.

        Args:
            name: The name of the dataset.
            sizes: The sizes of the dimensions.
            horizontal_coord_unit: The units for the horizontal coordinates.
            headers: Optional structured headers for the dataset.

        Returns:
            Dataset: The constructed dataset
        """
        self._dim_sizes = sizes
        self._horizontal_coord_unit = horizontal_coord_unit

        self._builder = MDIODatasetBuilder(name=name, attributes=self._load_dataset_attributes())
        self._add_dimensions()
        self._add_coordinates()
        self._add_variables()
        self._add_trace_mask()
        if headers:
            self._add_trace_headers(headers)
        return self._builder.build()

    @property
    def name(self) -> str:
        """Returns the name of the template."""
        return self._name

    @property
    def trace_variable_name(self) -> str:
        """Returns the name of the trace variable."""
        return self._trace_variable_name

    @property
    def trace_domain(self) -> str:
        """Returns the name of the trace domain."""
        return self._trace_domain

    @property
    def dimension_names(self) -> list[str]:
        """Returns the names of the dimensions."""
        return copy.deepcopy(self._dim_names)

    @property
    def coordinate_names(self) -> list[str]:
        """Returns the names of the coordinates."""
        return copy.deepcopy(self._coord_names)

    @property
    @abstractmethod
    def _name(self) -> str:
        """Abstract method to get the name of the template.

        Must be implemented by subclasses.

        Returns:
            str: The name of the template
        """

    @property
    def _trace_variable_name(self) -> str:
        """Get the name of the data variable.

        A virtual method that can be overwritten by subclasses to return a
        custom data variable name.

        Returns:
            str: The name of the data variable
        """
        return "amplitude"

    @abstractmethod
    def _load_dataset_attributes(self) -> UserAttributes:
        """Abstract method to load dataset attributes.

        Must be implemented by subclasses.

        Returns:
            UserAttributes: The dataset attributes
        """

    def _add_dimensions(self) -> None:
        """Add custom dimensions.

        A virtual method that can be overwritten by subclasses to add custom dimensions.
        Uses the class field 'builder' to add dimensions to the dataset.
        """
        for i in range(len(self._dim_names)):
            self._builder.add_dimension(self._dim_names[i], self._dim_sizes[i])

    def _add_coordinates(self) -> None:
        """Add custom coordinates.

        A virtual method that can be overwritten by subclasses to add custom coordinates.
        Uses the class field 'builder' to add coordinates to the dataset.
        """
        # Add dimension coordinates
        for name in self._dim_names:
            self._builder.add_coordinate(
                name,
                dimensions=[name],
                data_type=ScalarType.INT32,
                metadata_info=None,
            )

        # Add non-dimension coordinates
        # TODO(Dmitriy Repin): do chunked write for non-dimensional coordinates and trace_mask
        # https://github.com/TGSAI/mdio-python/issues/587
        # The chunk size used for trace mask will be different from the _var_chunk_shape
        hor_coord_units = [self._horizontal_coord_unit] * len(self._coord_dim_names)
        for i in range(len(self._coord_names)):
            self._builder.add_coordinate(
                self._coord_names[i],
                dimensions=self._coord_dim_names,
                data_type=ScalarType.FLOAT64,
                metadata_info=hor_coord_units,
            )

    def _add_trace_mask(self) -> None:
        """Add trace mask variables."""
        # TODO(Dmitriy Repin): do chunked write for non-dimensional coordinates and trace_mask
        # https://github.com/TGSAI/mdio-python/issues/587
        # The chunk size used for trace mask will be different from the _var_chunk_shape
        self._builder.add_variable(
            name="trace_mask",
            dimensions=self._dim_names[:-1],  # All dimensions except vertical (the last one)
            data_type=ScalarType.BOOL,
            compressor=compressors.Blosc(algorithm=compressors.BloscAlgorithm.ZSTD),
            coordinates=self._coord_names,
            metadata_info=None,
        )

    def _add_trace_headers(self, headers: StructuredType) -> None:
        """Add trace mask variables."""
        # headers = StructuredType.model_validate(header_fields)

        self._builder.add_variable(
            name="headers",
            dimensions=self._dim_names[:-1],  # All dimensions except vertical (the last one)
            data_type=headers,
            compressor=compressors.Blosc(algorithm=compressors.BloscAlgorithm.ZSTD),
            coordinates=self._coord_names,
            metadata_info=[
                ChunkGridMetadata(
                    chunk_grid=RegularChunkGrid(
                        configuration=RegularChunkShape(chunk_shape=self._var_chunk_shape[:-1])
                    )
                )
            ],
        )

    def _add_variables(self) -> None:
        """Add custom variables.

        A virtual method that can be overwritten by subclasses to add custom variables.
        Uses the class field 'builder' to add variables to the dataset.
        """
        self._builder.add_variable(
            name=self._trace_variable_name,
            dimensions=self._dim_names,
            data_type=ScalarType.FLOAT32,
            compressor=compressors.Blosc(algorithm=compressors.BloscAlgorithm.ZSTD),
            coordinates=self._coord_names,
            metadata_info=[
                ChunkGridMetadata(
                    chunk_grid=RegularChunkGrid(
                        configuration=RegularChunkShape(chunk_shape=self._var_chunk_shape)
                    )
                )
            ],
        )
