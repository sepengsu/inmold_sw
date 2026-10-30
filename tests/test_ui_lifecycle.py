"""Exercise browser disconnects against a real HTTP server without hardware."""

import asyncio
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock

import httpx
import uvicorn

from src.web.app import create_app
from src.web.service import Workbench
from src.web.ui_lifecycle import UiSessions


ROOT = Path(__file__).resolve().parents[1]


class UiLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.workbench = Workbench(
            config_path=ROOT / "config/preprocess.yaml", checkpoint_path=ROOT / "model/missing.pt",
            device="cpu", batch_size=8, camera_serial=None, use_camera=False,
            camera_config_path=ROOT / "config/camera.yaml", history_root=Path(self.folder.name),
            load_model=False, auto_connect_light=False, result_sender=Mock(),
        )
        self.stop = Mock(wraps=self.workbench.stop)
        self.workbench.stop = self.stop
        self.shutdown = Mock(side_effect=lambda: setattr(self.server, "should_exit", True))
        sessions = UiSessions(self.shutdown, lambda: self.server.should_exit, grace_seconds=0.15)
        app = create_app(self.workbench, ui_sessions=sessions)
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.addCleanup(self.socket.close)
        port = self.socket.getsockname()[1]
        self.server_task = asyncio.create_task(self.server.serve(sockets=[self.socket]))
        self.addAsyncCleanup(self.finish_server)

        async def wait_for_start():
            while not self.server.started:
                if self.server_task.done():
                    await self.server_task
                    self.fail("Server exited before startup.")
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_start(), timeout=5)
        self.client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=3)
        self.addAsyncCleanup(self.client.aclose)

    async def finish_server(self):
        self.server.should_exit = True
        await asyncio.wait_for(self.server_task, timeout=5)

    async def open_ui(self):
        response = await self.client.send(self.client.build_request("GET", "/api/ui/session"), stream=True)
        self.addAsyncCleanup(response.aclose)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        return response

    async def test_unopened_ui_does_not_stop_headless_server(self):
        await asyncio.sleep(0.3)
        self.assertFalse(self.server_task.done())
        self.shutdown.assert_not_called()

    async def test_last_ui_disconnect_stops_server_and_releases_resources(self):
        response = await self.open_ui()
        await response.aclose()
        await asyncio.wait_for(self.server_task, timeout=3)
        self.shutdown.assert_called_once()
        self.stop.assert_called_once()

    async def test_refresh_reconnect_cancels_shutdown(self):
        first = await self.open_ui()
        await first.aclose()
        await asyncio.sleep(0.04)
        replacement = await self.open_ui()
        await asyncio.sleep(0.3)
        self.assertFalse(self.server_task.done())
        self.shutdown.assert_not_called()
        await replacement.aclose()
        await asyncio.wait_for(self.server_task, timeout=3)
        self.stop.assert_called_once()

    async def test_other_open_tab_keeps_server_running(self):
        first = await self.open_ui()
        second = await self.open_ui()
        await first.aclose()
        await asyncio.sleep(0.3)
        self.assertFalse(self.server_task.done())
        self.shutdown.assert_not_called()
        await second.aclose()
        await asyncio.wait_for(self.server_task, timeout=3)
        self.stop.assert_called_once()

    async def test_manual_shutdown_finishes_even_with_ui_still_open(self):
        response = await self.open_ui()
        self.server.should_exit = True
        body = await response.aread()
        self.assertIn(b"event: shutdown", body)
        await asyncio.wait_for(self.server_task, timeout=3)
        self.shutdown.assert_not_called()
        self.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
