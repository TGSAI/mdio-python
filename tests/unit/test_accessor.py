"""Test for MDIO accessors."""

import os
import shutil

import numpy as np
import numpy.testing as npt
import pytest

from mdio import MDIOReader
from mdio.core.exceptions import MDIOAlreadyExistsError
from mdio.core.exceptions import MDIONotFoundError
from mdio.exceptions import ShapeError
from mdio.seismic.helpers_segy import create_zarr_hierarchy


class TestReader:
    """Tests for reader units."""

    def test_basic_attrs(self, mock_reader, mock_data):
        """Compare ingested basic attrs to original."""
        assert mock_reader.n_dim == mock_data.ndim
        assert mock_reader.trace_count == np.prod(mock_data.shape[:-1])

    def test_basic_stats(self, mock_reader, mock_data):
        """Ensure access to stats work properly."""
        assert mock_reader.stats["mean"] == mock_data.mean()
        assert mock_reader.stats["std"] == mock_data.std()
        assert mock_reader.stats["min"] == mock_data.min()
        assert mock_reader.stats["max"] == mock_data.max()

    def test_text_hdr(self, mock_reader, mock_text):
        """Compare ingested text header to original."""
        assert mock_reader.text_header == mock_text

    def test_bin_hdr(self, mock_reader, mock_bin):
        """Compare ingested binary header to original."""
        assert mock_reader.binary_header == mock_bin

    def test_shape(self, mock_reader, mock_data):
        """Compare ingested shape to expected."""
        assert mock_reader.shape == mock_data.shape
        assert mock_reader.chunks == mock_data.shape

    def test_live_mask(self, mock_reader):
        """Check if live mask is full as expected."""
        assert np.all(mock_reader.live_mask[:])

    @pytest.mark.parametrize(
        "il_coord, il_index, xl_coord, xl_index, z_coord, z_index",
        [
            (101, 0, 10, 0, 0, 0),
            (115, 7, 15, 5, 50, 10),
            (129, 14, 19, 9, 95, 19),
            ([101, 115, 129], [0, 7, 14], 11, 1, 10, 2),
            ([101, 129], [0, 14], 11, 1, [10, 95], [2, 19]),
            ([101], [0], [11], [1], [95], [19]),
        ],
    )
    def test_coord_slicing(
        self,
        il_coord,
        il_index,
        xl_coord,
        xl_index,
        z_coord,
        z_index,
        mock_reader,
        mock_data,
    ):
        """Test IL/XL number to Index slicing."""
        il_indices = mock_reader.coord_to_index(il_coord, dimensions="inline")
        xl_indices = mock_reader.coord_to_index(xl_coord, dimensions="crossline")
        z_indices = mock_reader.coord_to_index(z_coord, dimensions="sample")

        # 2-D should work too
        _ = mock_reader.coord_to_index(
            il_coord,
            xl_coord,
            dimensions=["inline", "crossline"],
        )

        # All dims should also work without specifying
        _ = mock_reader.coord_to_index(il_coord, xl_coord, z_coord)

        il_indices = np.atleast_1d(il_indices)
        il_index = np.atleast_1d(il_index)
        xl_indices = np.atleast_1d(xl_indices)
        xl_index = np.atleast_1d(xl_index)
        z_indices = np.atleast_1d(z_indices)
        z_index = np.atleast_1d(z_index)

        npt.assert_array_equal(il_indices, il_index)
        npt.assert_array_equal(xl_indices, xl_index)
        npt.assert_array_equal(z_indices, z_index)

        # TODO: Add strict=True and remove noqa when minimum Python is 3.10
        for act_idx, exp_idx in zip(il_indices, il_index):  # noqa: B905
            npt.assert_array_equal(mock_reader[act_idx], mock_data[exp_idx])

        # TODO: Add strict=True and remove noqa when minimum Python is 3.10
        for act_idx, exp_idx in zip(xl_indices, xl_index):  # noqa: B905
            npt.assert_array_equal(mock_reader[:, act_idx], mock_data[:, exp_idx])

        # TODO: Add strict=True and remove noqa when minimum Python is 3.10
        for act_idx, exp_idx in zip(z_indices, z_index):  # noqa: B905
            npt.assert_array_equal(mock_reader[..., act_idx], mock_data[..., exp_idx])

    def test_local_caching(self, mock_reader_cached):
        """Test local caching."""
        assert os.path.isdir("./mdio_test_cache")
        shutil.rmtree("./mdio_test_cache")


class TestExceptions:
    """Test custom exceptions and if they're raised properly."""

    def test_mdio_not_found(self) -> None:
        """MDIO doesn't exist or corrupt."""
        with pytest.raises(MDIONotFoundError):
            MDIOReader("prefix/file_that_doesnt_exist.mdio")

    def test_wrong_size_index(self, mock_reader: MDIOReader) -> None:
        """If user asks for N dimensions but didn't specify all."""
        with pytest.raises(ShapeError):
            mock_reader.coord_to_index(0, 0, dimensions="inline")

    def test_wrong_index(self, mock_reader: MDIOReader) -> None:
        """If user asks for an index that doesn't exist."""
        with pytest.raises(ValueError):
            mock_reader.coord_to_index(0, dimensions="non_existent")

    def test_mdio_exists(self, mock_reader: MDIOReader) -> None:
        """MDIO doesn't exist or corrupt."""
        mock_store = mock_reader.store
        with pytest.raises(MDIOAlreadyExistsError):
            create_zarr_hierarchy(mock_store, overwrite=False)
