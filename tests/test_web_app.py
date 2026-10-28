"""FastAPI workbench behavior without camera or serial hardware."""

import tempfile
import unittest
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.hardware.camera_config import load_camera_settings
from src.hardware.camera_control import CameraControl
from src.hardware.lighting_config import LightingConfigStore
from src.hardware.led_control import LightController
from src.inspection_history import InspectionHistoryStore
from src.roi.threshold_config import RoiThresholdConfigStore
from src.web.app import create_app
from src.web.service import Workbench


ROOT = Path(__file__).resolve().parents[1]


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        workbench = Workbench.__new__(Workbench)
        workbench.lock = RLock()
        workbench.camera_enabled = False
        workbench.auto_connect_light = False
        workbench.camera_config_path = ROOT / "config/camera.yaml"
        workbench.camera = CameraControl(settings=load_camera_settings(workbench.camera_config_path))
        workbench.light = LightController()
        workbench.history = InspectionHistoryStore(self.folder.name)
        workbench.lighting_config = LightingConfigStore(ROOT / "config/lighting.yaml")
        workbench.roi_config = RoiThresholdConfigStore(ROOT / "config/roi_gabor_boundary.yaml")
        workbench.model_error = None
        frame = np.full((12, 15, 3), 80, dtype=np.uint8)
        inference = SimpleNamespace(
            scores=(90.0, 80.0, 70.0), average=80.0, annotated_frame=frame,
            crops=(frame[:4, :4], frame[:4, 4:8], frame[:4, 8:12]),
            regions=tuple({"x": x, "y": 0, "width": 4, "height": 4} for x in (0, 4, 8)),
            roi=None, roi_detection=None,
        )
        workbench.engine = SimpleNamespace(
            config={"roi": {"x": 0, "y": 0, "width": 15, "height": 12},
                    "auto_roi": {"enabled": False}},
            last_roi_detection=None, checkpoint={"threshold": 0.5,
                                                  "positive_classes": ("Exposure", "Weldline")},
            model_info={"threshold": 0.5, "positive_classes": ("Exposure", "Weldline")},
            inspect=lambda pixels: inference,
            inspect_certification=lambda pixels: inference,
        )
        self.workbench = workbench
        self.client = TestClient(create_app(workbench))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.assertTrue(ok)
        self.image = encoded.tobytes()

    def test_html_bootstrap_and_roi_preview(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("측정 시작", page.text)
        self.assertIn("/static/app.js", page.text)
        bootstrap = self.client.get("/api/bootstrap").json()
        self.assertEqual(bootstrap["source"], "file")
        self.assertEqual(bootstrap["status"]["positive_classes"], ["Exposure", "Weldline"])
        self.assertEqual(len(bootstrap["lighting"]["groups"]), 4)
        preview = self.client.get("/api/roi/preview", params={"level": "low"}).json()
        self.assertEqual(preview["threshold"], 0.25)
        self.assertEqual(preview["level"], "low")

    def test_file_inspection_history_detail_export_and_clear(self):
        response = self.client.post("/api/inspect", data={"source": "file", "mode": "normal",
                                                          "lighting": "0,0,0,0"},
                                    files={"image": ("frame.png", self.image, "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["record"]["average"], 80.0)
        self.assertTrue(result["images"]["p2"].startswith("data:image/png;base64,"))
        record_id = result["record"]["id"]
        self.assertTrue(record_id)
        self.assertEqual(len(self.client.get("/api/history").json()), 1)
        statistics = self.client.get("/api/statistics").json()
        self.assertEqual(statistics["summary"]["count"], 1)
        self.assertEqual(statistics["rows"][0]["id"], record_id)
        detail = self.client.get(f"/api/statistics/{record_id}").json()
        self.assertEqual(set(detail["images"]), {"original", "result", "p1", "p2", "p3"})
        csv = self.client.get("/api/statistics/export/csv")
        self.assertEqual(csv.status_code, 200)
        self.assertIn("P2 함침율", csv.content.decode("utf-8-sig"))
        self.assertEqual(self.client.post("/api/statistics/clear", json={"confirmed": False}).status_code, 400)
        cleared = self.client.post("/api/statistics/clear", json={"confirmed": True})
        self.assertEqual(cleared.json()["count"], 1)
        self.assertEqual(self.client.get("/api/statistics").json()["summary"]["count"], 0)

    def test_missing_file_reports_error_without_history(self):
        response = self.client.post("/api/inspect", data={"source": "file", "mode": "normal"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/statistics").json()["summary"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
