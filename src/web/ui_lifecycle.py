"""Keep the desktop server alive while a browser UI is connected."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
import logging

from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send


class UiSessions:
    def __init__(self, request_shutdown: Callable[[], None],
                 server_stopping: Callable[[], bool], *, grace_seconds: float = 5.0):
        self.request_shutdown = request_shutdown
        self.server_stopping = server_stopping
        self.grace_seconds = grace_seconds
        self._connections = 0
        self._pending_shutdown: asyncio.Task | None = None
        self._closed = False

    @property
    def stopping(self) -> bool:
        return self._closed or self.server_stopping()

    def connect(self) -> None:
        self._connections += 1
        if self._pending_shutdown is not None:
            self._pending_shutdown.cancel()
            self._pending_shutdown = None

    def disconnect(self) -> None:
        self._connections -= 1
        if self._connections == 0 and not self.stopping:
            self._pending_shutdown = asyncio.create_task(self._shutdown_when_empty())

    async def _shutdown_when_empty(self) -> None:
        await asyncio.sleep(self.grace_seconds)
        if self._connections == 0 and not self.stopping:
            logging.getLogger(__name__).info("All inspection UI windows closed; stopping server.")
            self.request_shutdown()

    async def close(self) -> None:
        self._closed = True
        if self._pending_shutdown is not None:
            self._pending_shutdown.cancel()
            with suppress(asyncio.CancelledError):
                await self._pending_shutdown
            self._pending_shutdown = None

    async def events(self) -> AsyncIterator[str]:
        # A persistent connection also detects a browser process exiting without pagehide.
        yield "retry: 1000\ndata: connected\n\n"
        while not self.stopping:
            await asyncio.sleep(0.5)
            yield ": keep-alive\n\n"
        yield "event: shutdown\ndata: {}\n\n"


class UiSessionResponse(StreamingResponse):
    def __init__(self, sessions: UiSessions):
        super().__init__(sessions.events(), media_type="text/event-stream",
                         headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        self.sessions = sessions

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.sessions.connect()
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.sessions.disconnect()
