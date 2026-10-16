"""Persist shared lighting values without transmitting them to the controller."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from Cam_test import CamTestController, build_ui
from src.hardware.led_control import LIGHT_GROUPS, LightConnectionStatus
from src.hardware.lighting_config import LightingConfigStore


class LightingConfigTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "lighting.yaml"
        self.store = LightingConfigStore(self.path)

    def test_new_store_loads_saved_values_by_fixture_key_not_yaml_order(self):
        self.store.save((11, 22, 33, 44))
        data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        data["brightness_percent"] = dict(reversed(list(data["brightness_percent"].items())))
        self.path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        self.assertEqual(LightingConfigStore(self.path).load(), (11, 22, 33, 44))

    def test_invalid_save_never_replaces_previous_file(self):
        self.store.save((11, 22, 33, 44))
        original = self.path.read_bytes()
        invalid_values = (-1, 101, float("nan"), float("inf"), True, "40", None, 10 ** 400)
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                self.assertIn("저장 실패", self.store.save_for_ui((11, 22, 33, invalid)))
                self.assertEqual(self.path.read_bytes(), original)
        self.assertIn("저장 실패", self.store.save_for_ui((11, 22, 33)))
        self.assertEqual(self.path.read_bytes(), original)

    def test_failed_atomic_replace_preserves_existing_settings_and_cleans_temp(self):
        self.store.save((11, 22, 33, 44))
        with patch("src.hardware.lighting_config.os.replace", side_effect=PermissionError("file locked")):
            message = self.store.save_for_ui((21, 32, 43, 54))
        self.assertIn("저장 실패", message)
        self.assertIn("file locked", message)
        self.assertEqual(self.store.load(), (11, 22, 33, 44))
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_invalid_file_is_reported_and_never_overwritten_by_load(self):
        self.store.save((11, 22, 33, 44))
        valid = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        bad_documents = ["[invalid", "null", "[]", "version: 2\nbrightness_percent: {}",
                         "version: true\nbrightness_percent: {}", "version: 1\nbrightness_percent: {}"]
        for invalid in (float("nan"), True, "40", -1, 101):
            data = {"version": 1, "brightness_percent": {**valid["brightness_percent"], "left_lower_panel": invalid}}
            bad_documents.append(yaml.safe_dump(data))
        for text in bad_documents:
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                values, message = self.store.load_for_ui()
                self.assertIsNone(values)
                self.assertIn("불러오기 실패", message)
                self.assertEqual(self.path.read_text(encoding="utf-8"), text)

    def test_missing_file_does_not_get_created_by_initial_load(self):
        values, message = self.store.initial_for_ui()
        self.assertEqual(values, (0,) * 4)
        self.assertIn("불러오기 실패", message)
        self.assertIn("기본값 0%", message)
        self.assertFalse(self.path.exists())


class LightingConfigUiTests(unittest.TestCase):
    def setUp(self):
        from src.gui import app

        self.app = app
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = LightingConfigStore(Path(folder.name) / "lighting.yaml")
        self.store.save((11, 22, 33, 44))
        main_store_patch = patch.object(app, "lighting_config", self.store)
        main_store_patch.start()
        self.addCleanup(main_store_patch.stop)
        serial_patch = patch("src.hardware.led_control.serial.Serial")
        self.serial_factory = serial_patch.start()
        self.addCleanup(serial_patch.stop)
        self.controller = CamTestController(lighting_config=self.store.path)
        self.controller.light = Mock(port="COM3", baudrate=9600)
        self.controller.light.get_status.return_value = LightConnectionStatus(False, False, None)
        self.controller.camera = Mock()
        self.controller.camera.get_status.return_value = type("Status", (), {
            "connected": False, "running": False, "last_frame_at": None,
            "error": None, "description": "Test camera",
        })()
        with patch("Cam_test.list_ports.comports", return_value=[]):
            self.cam_demo = build_ui(self.controller)
        self.cam_callbacks = {fn.api_name: fn.fn for fn in self.cam_demo.fns.values()}
        self.main_callbacks = {fn.api_name: fn.fn for fn in app.demo.fns.values()}

    def assert_no_light_transmission(self):
        self.controller.light.send.assert_not_called()
        self.serial_factory.assert_not_called()

    def test_main_save_can_be_loaded_in_camera_and_camera_save_in_main(self):
        message = self.main_callbacks["lighting_config_save"](12, 34, 56, 78)
        self.assertIn("저장 완료", message)
        self.assertEqual(self.cam_callbacks["lighting_config_load"]()[:4], (12, 34, 56, 78))
        message = self.cam_callbacks["lighting_config_save"](21, 43, 65, 87)
        self.assertIn("저장 완료", message)
        loaded = self.main_callbacks["lighting_config_load"]()
        self.assertEqual(loaded[:4], (21, 43, 65, 87))
        self.assertEqual(loaded[4], 54)
        self.assertIn("54%", loaded[5])
        self.assert_no_light_transmission()

    def test_startup_and_page_load_use_latest_config_without_transmitting(self):
        sliders = [block for block in self.cam_demo.blocks.values()
                   if getattr(block, "label", "") in {f"{group.name} 밝기 (%)" for group in LIGHT_GROUPS}]
        self.assertEqual([slider.value for slider in sliders], [11, 22, 33, 44])
        self.store.save((9, 19, 29, 39))
        main_load = next(fn.fn for fn in self.app.demo.fns.values() if fn.fn is self.app.loadLightingConfig)
        cam_load = next(fn.fn for fn in self.cam_demo.fns.values() if getattr(fn.fn, "__name__", "") == "load_lighting_config")
        self.assertEqual(main_load()[:4], (9, 19, 29, 39))
        self.assertEqual(cam_load()[:4], (9, 19, 29, 39))
        self.controller.start()
        self.controller.camera.start.assert_called_once()
        self.controller.light.connect.assert_called_once()
        self.assert_no_light_transmission()

    def test_reload_failure_keeps_existing_sliders_and_mean(self):
        self.store.path.write_text("version: 1\nbrightness_percent: {}", encoding="utf-8")
        main_result = self.main_callbacks["lighting_config_load"]()
        cam_result = self.cam_callbacks["lighting_config_load"]()
        self.assertEqual(main_result[:6], ({"__type__": "update"},) * 6)
        self.assertEqual(cam_result[:4], ({"__type__": "update"},) * 4)
        self.assertIn("불러오기 실패", main_result[-1])
        self.assertIn("불러오기 실패", cam_result[-1])
        self.assert_no_light_transmission()

    def test_save_failure_is_visible_in_both_uis_and_keeps_saved_values(self):
        with patch("src.hardware.lighting_config.os.replace", side_effect=PermissionError("file locked")):
            for callback in (self.main_callbacks["lighting_config_save"], self.cam_callbacks["lighting_config_save"]):
                with self.subTest(callback=callback):
                    result = callback(12, 34, 56, 78)
                    self.assertIn("저장 실패", result)
                    self.assertEqual(self.store.load(), (11, 22, 33, 44))
        self.assert_no_light_transmission()

    def test_main_slider_release_is_not_wired_to_hardware(self):
        slider_ids = {slider._id for slider in self.app.light_controls}
        releases = [target for dependency in self.app.demo.config["dependencies"]
                    for target in dependency["targets"] if target[0] in slider_ids and target[1] == "release"]
        self.assertEqual(releases, [])
        self.assert_no_light_transmission()


if __name__ == "__main__":
    unittest.main()
