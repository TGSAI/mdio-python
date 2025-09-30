"""Consumer-side utility to get both raw and transformed header data with single filesystem read."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from segy import SegyFile


class SegyFileRawTraceWrapper:
    def __init__(self, segy_file: SegyFile, indices: int | list[int] | NDArray | slice):
        self.segy_file = segy_file
        self.indices = indices

        self.idx = self.segy_file.trace.normalize_and_validate_query(self.indices)
        self.trace_buffer_array = self.segy_file.trace.fetch(self.idx, raw=True)

        self.trace_view = self.trace_buffer_array.view(self.segy_file.spec.trace.dtype)

        self.trace_decode_pipeline = self.segy_file.accessors.trace_decode_pipeline
        self.decoded_traces = None  # decode later when not-raw header/sample is called

    def _ensure_decoded(self) -> None:
        """Apply trace decoding pipeline if not already done."""
        if self.decoded_traces is not None:  # already done
            return
        self.decoded_traces = self.trace_decode_pipeline.apply(self.trace_view.copy())

    @property
    def raw_header(self) -> NDArray:
        """Get byte array view of the raw headers."""
        header_itemsize = self.segy_file.spec.trace.header.itemsize  # should be 240
        return self.trace_view.header.view(np.dtype((np.void, header_itemsize)))

    @property
    def header(self) -> NDArray:
        """Get decoded header."""
        self._ensure_decoded()  # decode when needed in-place to avoid copy.
        return self.decoded_traces.header

    @property
    def sample(self) -> NDArray:
        """Get decoded trace samples."""
        self._ensure_decoded()  # decode when needed in-place to avoid copy.
        return self.decoded_traces.sample
