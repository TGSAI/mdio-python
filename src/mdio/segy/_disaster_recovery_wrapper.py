"""Consumer-side utility to get both raw and transformed header data with single filesystem read."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from segy import SegyFile


class SegyFileTraceDataWrapper:
    def __init__(self, segy_file: SegyFile, indices: int | list[int] | NDArray | slice):
        self.segy_file = segy_file
        self.indices = indices

        self.idx = self.segy_file.trace.normalize_and_validate_query(self.indices)
        self.traces = self.segy_file.trace.fetch(self.idx, raw=True)

        self.raw_view = self.traces.view(self.segy_file.spec.trace.dtype)
        self.decoded_traces = self.segy_file.accessors.trace_decode_pipeline.apply(self.raw_view.copy())

    @property
    def raw_header(self) -> NDArray:
        return self.raw_view.header.view("|V240")

    @property
    def header(self) -> NDArray:
        return self.decoded_traces.header

    @property
    def sample(self) -> NDArray:
        return self.decoded_traces.sample
