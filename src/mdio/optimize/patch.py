"""Dask worker plugins for monkey patching ZFP due to bug.

We can remove this once the fix is upstreamed:
https://github.com/zarr-developers/numcodecs/issues/812
https://github.com/zarr-developers/numcodecs/pull/811
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
from numcodecs import blosc
from zarr.codecs import numcodecs

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer
    from zarr.core.buffer import NDBuffer


try:
    import distributed
except ModuleNotFoundError:
    distributed = None


class ZFPY(numcodecs.ZFPY, codec_name="zfpy"):
    """Monkey patch ZFP codec to make input array contiguous before encoding."""

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        chunk_ndarray = chunk_data.as_ndarray_like()
        if not chunk_ndarray.flags.c_contiguous:
            chunk_ndarray = np.ascontiguousarray(chunk_ndarray)
        out = await asyncio.to_thread(self._codec.encode, chunk_ndarray)
        return chunk_spec.prototype.buffer.from_bytes(out)


if distributed is not None:

    class MonkeyPatchZfpDaskPlugin(distributed.WorkerPlugin):
        """Monkey patch ZFP codec and disable Blosc threading for Dask workers.

        Note that this is class is only importable if distributed is installed. However, in the caller
        function we have a context manager that checks if distributed is installed, so it is safe (for now).
        """

        def setup(self, worker: distributed.Worker) -> None:  # noqa: ARG002
            """Monkey patch ZFP codec and disable Blosc threading."""
            numcodecs._codecs.ZFPY = ZFPY
            blosc.set_nthreads(1)
