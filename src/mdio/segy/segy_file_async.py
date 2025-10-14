"""SEG-Y async file support and utilities."""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import threading
from typing import TYPE_CHECKING
from typing import TypedDict

import fsspec
from fsspec.asyn import AsyncFileSystem
from fsspec.utils import get_protocol
from segy import SegyFile
from segy.config import SegyFileSettings

if TYPE_CHECKING:
    from pathlib import Path

    from segy.config import SegyHeaderOverrides
    from segy.schema.segy import SegySpec


__all__ = ["SegyFileArguments", "SegyFileAsync", "MDIO_ASYNCIO_THREAD_STOP_TIMEOUT"]

logger = logging.getLogger(__name__)

# Timeout in seconds for stopping async event loop threads during cleanup
MDIO_ASYNCIO_THREAD_STOP_TIMEOUT = 5.0


def _start_asyncio_loop(segy_file_kwargs: SegyFileArguments) -> None:
    """Start asyncio event loop for async filesystems.

    If the filesystem is async (e.g., S3, GCS, Azure), creates a new event loop
    in a daemon thread and injects it into the storage options.

    Args:
        segy_file_kwargs: SEG-Y file arguments that will be modified to include the loop.
    """
    protocol = get_protocol(str(segy_file_kwargs["url"]))
    # Get the filesystem class without instantiating it
    fs_class = fsspec.get_filesystem_class(protocol)
    # Only create event loop for async filesystems
    is_async = issubclass(fs_class, AsyncFileSystem)
    if not is_async:
        return

    # Create a new event loop and thread to run it in a daemon thread.
    loop_asyncio = asyncio.new_event_loop()
    th_asyncio = threading.Thread(
        target=loop_asyncio.run_forever,
        name=f"mdio-{os.getpid()}",
        daemon=True,
    )
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
    if loop_asyncio.is_running():
        loop_asyncio.call_soon_threadsafe(loop_asyncio.stop)

    th_asyncio.join(timeout=MDIO_ASYNCIO_THREAD_STOP_TIMEOUT)

    if th_asyncio.is_alive():
        # Thread did not terminate within timeout, but daemon threads will be
        # terminated by Python interpreter on exit anyway
        logger.warning(
            "Async event loop thread '%s' did not terminate within %s seconds",
            th_asyncio.name,
            MDIO_ASYNCIO_THREAD_STOP_TIMEOUT,
        )


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
