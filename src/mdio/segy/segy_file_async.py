"""SEG-Y async file support and utilities."""

from __future__ import annotations

import asyncio
import atexit
import os
import threading
from typing import TYPE_CHECKING
from typing import TypedDict
from urllib.parse import urlparse

import fsspec
from fsspec.asyn import AsyncFileSystem
from segy import SegyFile
from segy.config import SegyFileSettings

if TYPE_CHECKING:
    from pathlib import Path

    from segy.config import SegyHeaderOverrides
    from segy.schema.segy import SegySpec


__all__ = ["SegyFileArguments", "SegyFileAsync"]


def _start_asyncio_loop(segy_file_kwargs: SegyFileArguments) -> None:
    """Start asyncio event loop for async filesystems.

    If the filesystem is async (e.g., S3, GCS), creates a new event loop
    in a daemon thread and injects it into the storage options.

    Args:
        segy_file_kwargs: SEG-Y file arguments that will be modified to include the loop.
    """
    # Get the filesystem class without instantiating it

    fs_class = fsspec.get_filesystem_class(urlparse(str(segy_file_kwargs["url"])).scheme)
    is_async = issubclass(fs_class, AsyncFileSystem)
    if is_async:
        # Create a new event loop and thread to run it in a daemon thread.
        loop_asyncio = asyncio.new_event_loop()
        th_asyncio = threading.Thread(target=loop_asyncio.run_forever, name=f"mdio-{os.getpid()}")
        th_asyncio.daemon = True
        th_asyncio.start()

        # Add the loop to the storage options to pass as a parameter to AsyncFileSystem.
        # Create a new settings object to avoid modifying the original (which may be shared).
        old_settings = segy_file_kwargs.get("settings") or SegyFileSettings()
        storage_options = {**(old_settings.storage_options or {}), "loop": loop_asyncio}
        segy_file_kwargs["settings"] = SegyFileSettings(
            endianness=old_settings.endianness,
            storage_options=storage_options,
        )

        # Register a function to stop the event loop and join the thread.
        atexit.register(_stop_asyncio_loop, loop_asyncio, th_asyncio)


def _stop_asyncio_loop(loop_asyncio: asyncio.AbstractEventLoop, th_asyncio: threading.Thread) -> None:
    """Stop the asyncio event loop and join the thread.

    Args:
        loop_asyncio: The asyncio event loop to stop.
        th_asyncio: The thread running the event loop.
    """
    loop_asyncio.stop()
    th_asyncio.join()


class SegyFileArguments(TypedDict):
    """Arguments to open SegyFile instance creation."""

    url: Path | str
    spec: SegySpec | None
    settings: SegyFileSettings | None
    header_overrides: SegyHeaderOverrides | None


class SegyFileAsync(SegyFile):
    """SEG-Y file that can be instantiated side by side with Zarr for cloud access.

    This is a workaround for Zarr issues 3487 'Explicitly using fsspec and zarr FsspecStore causes
    RuntimeError "Task attached to a different loop"'

    # TODO (Dmitriy Repin): when Zarr issue 3487 is resolved, we can remove this workaround
    # https://github.com/zarr-developers/zarr-python/issues/3487

    Args:
        url: Path to the SEG-Y file.
        spec: SEG-Y specification.
        settings: SEG-Y settings.
        header_overrides: SEG-Y header overrides.
    """

    def __init__(
        self,
        url: Path | str,
        spec: SegySpec | None = None,
        settings: SegyFileSettings | None = None,
        header_overrides: SegyHeaderOverrides | None = None,
    ):
        args = SegyFileArguments(
            url=url,
            spec=spec,
            settings=settings,
            header_overrides=header_overrides,
        )
        _start_asyncio_loop(args)
        super().__init__(**args)
