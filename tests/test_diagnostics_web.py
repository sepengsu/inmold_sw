"""FastAPI diagnostic pages and hardware actions with simulated devices."""

import tempfile
import unittest
from base64 import b64decode
from pathlib import Path
from time import monotonic
from unittest.mock import Mock, patch

import numpy as np
import cv2
from fastapi.testclient import TestClient

from src.hardware.camera_control import CameraFrame, CameraStatus
from src.hardware.led_control import ACK
from src.preprocess.auto_roi import AutoRoiResult
from src.web.app import create_app
from src.web.service import Workbench


ROOT = Path(__file__).resolve().parents[1]


class DiagnosticWebTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        config = self.root / "camera.yaml"
        config.write_bytes((ROOT / "config/camera.yaml").read_bytes())
        self.workbench = Workbench(
            config_path=ROOT / "config/preprocess.yaml",
            checkpoint_path=ROOT / "model/best_model.pt", device="cpu", batch_size=8,
            camera_serial=None, use_camera=False, camera_config_path=config,
            history_root=self.root / "history", load_model=False, auto_connect_light=False,
        )
        self.workbench.capture_dir = self.root / "captures"
        self.app = create_app(self.workbench)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_pages_bootstrap_and_no_camera_preview(self):
        self.assertIsNone(self.workbench.engine)
        for url in ("/camera-test", "/rs232-test"):
            self.assertEqual(self.client.get(url).status_code, 200)
        self.assertEqual(self.client.get("/api/camera-test/preview").json()["image"], None)
        self.assertIsNone(self.client.get("/api/camera-test/preview?include_roi=true").json()["roi_image"])
        self.assertEqual(self.client.get("/api/rs232-test/bootstrap").json()["channels"], 6)

    def test_camera_capture_download_and_exposure(self):
        rgb = np.full((8, 9, 3), 80, dtype=np.uint8)
        camera = Mock()
        camera.serial_number = "24346077"
        camera.get_status.return_value = CameraStatus(
            running=True, connected=True, description="Test camera", serial_number="24346077",
            frames_received=5, last_frame_at=monotonic(), error=None,
        )
        camera.get_latest_frame.return_value = rgb.copy()
        camera.get_latest_camera_frame.return_value = CameraFrame(
            display_rgb=rgb.copy(), raw_array=np.full((8, 9), 1000, np.uint16),
            width=9, height=8, pixel_format="BayerGR12", block_id=5,
            camera_timestamp=100, captured_at_ns=100,
        )
        camera.set_exposure_us.return_value = 25000.0
        self.app.state.camera_test.camera = camera
        self.workbench.camera = camera
        preview = self.client.get("/api/camera-test/preview").json()
        self.assertTrue(preview["image"].startswith("data:image/jpeg;base64,"))
        self.assertIn("uint16", preview["frame_info"])
        capture = self.client.post("/api/camera-test/capture", json={"name": "sample"}).json()
        self.assertEqual(len(capture["files"]), 2)
        for file in capture["files"]:
            self.assertEqual(self.client.get(file["url"]).status_code, 200)
        repeated = self.client.post("/api/camera-test/capture", json={"name": "sample"}).json()
        self.assertTrue(all("sample_1_" in file["name"] for file in repeated["files"]))
        exposure = self.client.post("/api/camera-test/exposure", json={"exposure_us": 25000}).json()
        self.assertEqual(exposure["exposure_us"], 25000)
        self.assertEqual(self.client.get("/api/camera-test/exposure").json()["exposure_us"], 25000)
        self.assertEqual(self.client.post("/api/camera-test/capture", json={"name": "../bad"}).json()["files"], [])

    def test_optional_live_roi_stream_updates_without_reextracting(self):
        camera = Mock()
        rgb = np.full((12, 14, 3), 80, np.uint8)
        camera.get_status.side_effect = lambda: CameraStatus(
            running=True, connected=True, description="Test camera", serial_number="24346077",
            frames_received=10, last_frame_at=monotonic(), error=None,
        )
        camera.get_latest_camera_frame.return_value = CameraFrame(
            display_rgb=rgb, raw_array=rgb[:, :, 0], width=14, height=12, pixel_format="Mono8",
            block_id=10, camera_timestamp=100, captured_at_ns=100,
        )
        self.app.state.camera_test.camera = camera
        self.app.state.camera_test.roi_result = AutoRoiResult(np.ones((12, 14), bool), {
            "roi_detected": True, "quality_status": "PASS", "source_shape": [12, 14],
            "top_y": [0] * 14, "bottom_y_exclusive": [12] * 14,
        })
        self.assertNotIn("roi_image", self.client.get("/api/camera-test/preview?show_roi=false").json())
        for value in (80, 170):
            camera.get_latest_frame.return_value = np.full_like(rgb, value)
            data = self.client.get("/api/camera-test/preview?show_roi=false&include_roi=true").json()
            decoded = cv2.imdecode(np.frombuffer(b64decode(data["roi_image"].split(",", 1)[1]), np.uint8),
                                   cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(decoded, camera.get_latest_frame.return_value)
            self.assertIn("실시간", data["roi_frame_info"])

    def test_rs232_apply_ack_history_off_and_disconnect(self):
        port = Mock(is_open=False)
        port.open.side_effect = lambda: setattr(port, "is_open", True)
        port.close.side_effect = lambda: setattr(port, "is_open", False)
        port.read.return_value = ACK
        with patch("src.hardware.led_control.serial.Serial", return_value=port):
            self.assertEqual(self.client.get("/api/rs232-test/bootstrap").json()["connected"], False)
            self.assertFalse(self.client.post("/api/rs232-test/apply", json={
                "port": "COM3", "baudrate": 9600, "values": [10], "channel": 1}).json()["connected"])
            port.write.assert_not_called()
            connected = self.client.post("/api/rs232-test/connect", json={"port": "COM3", "baudrate": 9600}).json()
            self.assertTrue(connected["connected"])
            port.write.assert_not_called()
            applied = self.client.post("/api/rs232-test/apply", json={
                "port": "COM3", "baudrate": 9600, "values": [10], "channel": 1}).json()
            self.assertEqual(applied["acknowledged"], [1])
            self.assertEqual(port.write.call_args.args[0], b"\x021026\x03")
            self.assertEqual(applied["history"][0][-1], "ACK 확인")
            off = self.client.post("/api/rs232-test/off", json={"port": "COM3", "baudrate": 9600}).json()
            self.assertEqual(off["reset_channels"], [1, 2, 3, 4, 5, 6])
            self.assertEqual(len(off["history"]), 7)
            cleared = self.client.post("/api/rs232-test/history/clear").json()
            self.assertEqual(cleared["history"], [])
            before = port.write.call_count
            disconnected = self.client.post("/api/rs232-test/disconnect").json()
            self.assertFalse(disconnected["connected"])
            self.assertEqual(port.write.call_count, before)


if __name__ == "__main__":
    unittest.main()
