"""Shared ROI threshold persistence and UI routing without device access."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from src.roi.roi_boundary_quality import DEFAULT_CONFIG, load_config
from src.roi.threshold_config import RoiThresholdConfigStore
from src.roi.quality_profiles import linked_roi_settings


class RoiThresholdConfigTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "roi.yaml"
        self.path.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
        self.store = RoiThresholdConfigStore(self.path)

    def test_save_changes_linked_quality_settings_and_preserves_unrelated_settings_and_comments(self):
        original = self.path.read_text(encoding="utf-8")
        expected = yaml.safe_load(original)
        for section, values in linked_roi_settings(.61).items():
            expected[section].update(values)
        self.assertEqual(self.store.save(.61), .61)
        saved = self.path.read_text(encoding="utf-8")
        self.assertEqual(yaml.safe_load(saved), expected)
        self.assertEqual([line for line in original.splitlines() if line.lstrip().startswith("#")],
                         [line for line in saved.splitlines() if line.lstrip().startswith("#")])
        self.assertEqual(load_config(self.path)["quality"]["min_score"], .61)

    def test_invalid_save_never_changes_existing_file(self):
        original = self.path.read_bytes()
        for value in (-.01, 1.01, True, None, "0.4", float("nan"), float("inf"), 10 ** 400):
            with self.subTest(value=value):
                result, message = self.store.save_for_ui(value)
                self.assertIsNone(result)
                self.assertIn("저장 실패", message)
                self.assertEqual(self.path.read_bytes(), original)

    def test_threshold_roundtrip_preserves_float_type_and_precision(self):
        for value in (0, 1, .123456789):
            with self.subTest(value=value):
                self.store.save(value)
                self.assertEqual(self.store.load(), value)
                self.assertIsInstance(yaml.safe_load(self.path.read_text(encoding="utf-8"))["quality"]["min_score"], float)

    def test_failed_atomic_replace_preserves_file_and_cleans_temporary_file(self):
        original = self.path.read_bytes()
        with patch("src.roi.threshold_config.os.replace", side_effect=PermissionError("file locked")):
            value, message = self.store.save_for_ui(.61)
        self.assertIsNone(value)
        self.assertIn("file locked", message)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_custom_sparse_override_gains_only_requested_threshold(self):
        for source in ("gabor: {wavelength: 12}\n", "quality: {min_inlier_ratio: 0.6}\n"):
            with self.subTest(source=source):
                self.path.write_text(source, encoding="utf-8")
                expected = yaml.safe_load(source)
                for section, values in linked_roi_settings(.61).items():
                    expected.setdefault(section, {}).update(values)
                self.store.save(.61)
                self.assertEqual(yaml.safe_load(self.path.read_text(encoding="utf-8")), expected)

    def test_invalid_load_and_invalid_existing_config_are_reported_without_rewriting(self):
        for text in ("[invalid", "[]", "quality: {min_inlier_ratio: -1}\n"):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                value, message = self.store.load_for_ui()
                self.assertIsNone(value)
                self.assertIn("불러오기 실패", message)
                value, message = self.store.save_for_ui(.61)
                self.assertIsNone(value)
                self.assertIn("저장 실패", message)
                self.assertEqual(self.path.read_text(encoding="utf-8"), text)


class RoiThresholdUiTests(unittest.TestCase):
    def setUp(self):
        from Cam_test import CamTestController, build_ui
        from src.gui import app
        from src.hardware.lighting_config import LightingConfigStore

        self.app = app
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "roi.yaml"
        self.path.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
        self.store = RoiThresholdConfigStore(self.path)
        self.store.save(.38)
        lighting = LightingConfigStore(Path(folder.name) / "lighting.yaml")
        lighting.save((11, 22, 33, 44))
        engine = SimpleNamespace(config={"auto_roi": {"enabled": True, "boundary_config": self.path}})
        engine_patch = patch.object(app, "inspection_engine", engine)
        engine_patch.start()
        self.addCleanup(engine_patch.stop)
        self.controller = CamTestController(roi_config=self.path, lighting_config=lighting.path)
        with patch("Cam_test.list_ports.comports", return_value=[]):
            self.cam_demo = build_ui(self.controller)
        self.cam_callbacks = {fn.api_name: fn.fn for fn in self.cam_demo.fns.values()}
        self.main_callbacks = {fn.api_name: fn.fn for fn in app.demo.fns.values()}

    def test_threshold_saved_by_either_ui_is_loaded_by_both_and_detector(self):
        before = load_config(self.path)
        value, message = self.cam_callbacks["roi_threshold_save"](.43)[:2]
        self.assertEqual(value, .43)
        self.assertIn("저장 완료", message)
        self.assertEqual(self.main_callbacks["roi_threshold_load"]()[0], .43)
        self.main_callbacks["roi_threshold_save"](.49)
        self.assertEqual(self.cam_callbacks["roi_threshold_load"]()[0], .49)
        expected = deepcopy(before)
        for section, values in linked_roi_settings(.49).items():
            expected[section].update(values)
        self.assertEqual(load_config(self.path), expected)
        self.assertFalse(self.controller.camera.is_running)
        self.assertFalse(self.controller.light.get_status().connected)

    def test_both_uis_keep_input_value_on_failure(self):
        original = self.path.read_bytes()
        for callbacks in (self.main_callbacks, self.cam_callbacks):
            value, message = callbacks["roi_threshold_save"](-1)[:2]
            self.assertEqual(value, {"__type__": "update"})
            self.assertIn("저장 실패", message)
        self.assertEqual(self.path.read_bytes(), original)
        self.path.write_text("quality: {min_score: invalid}\n", encoding="utf-8")
        for callbacks in (self.main_callbacks, self.cam_callbacks):
            value, message = callbacks["roi_threshold_load"]()[:2]
            self.assertEqual(value, {"__type__": "update"})
            self.assertIn("불러오기 실패", message)

    def test_stage_selection_previews_without_saving_and_saved_stage_is_shared(self):
        from src.roi.quality_profiles import ROI_QUALITY_PROFILES

        for name, profile in ROI_QUALITY_PROFILES.items():
            with self.subTest(name=name):
                before = self.path.read_bytes()
                score, details = self.cam_callbacks["roi_level_preview"](name)
                self.assertEqual(score, profile["quality"]["min_score"])
                self.assertIn(f"{profile['quality']['min_inlier_ratio']:.1%}", details)
                self.assertEqual(self.path.read_bytes(), before)
                result = self.cam_callbacks["roi_threshold_save"](score)
                self.assertEqual(result[2], name)
                self.assertEqual(self.main_callbacks["roi_threshold_load"]()[2], name)
                saved = load_config(self.path)
                for section, values in profile.items():
                    for key, value in values.items():
                        self.assertEqual(saved[section][key], value)
        self.assertFalse(self.controller.camera.is_running)
        self.assertFalse(self.controller.light.get_status().connected)

    def test_manual_threshold_preview_and_save_adjust_related_gates(self):
        before = self.path.read_bytes()
        level, details = self.main_callbacks["roi_threshold_preview"](.325)
        self.assertIsNone(level)
        self.assertIn("45.0%", details)
        self.assertEqual(self.path.read_bytes(), before)
        self.main_callbacks["roi_threshold_save"](.325)
        actual = load_config(self.path)["quality"]
        self.assertAlmostEqual(actual["min_inlier_ratio"], .45)
        self.assertAlmostEqual(actual["min_x_coverage"], .55)
        self.assertEqual(self.cam_callbacks["roi_threshold_load"]()[0], .325)

    def test_main_uses_custom_roi_config_even_when_model_cannot_load(self):
        preprocess = self.path.parent / "preprocess.yaml"
        preprocess.write_text("roi: {x: 0, y: 0, width: 8, height: 8}\nauto_roi: {enabled: true, boundary_config: roi.yaml}\n", encoding="utf-8")
        with patch.multiple(self.app, inspection_engine=None, model_error=None, camera_control=None,
                            camera_enabled=False, roi_config_path=None), \
                patch("src.gui.inspection.InspectionEngine", side_effect=RuntimeError("model unavailable")), \
                patch.object(self.app, "CameraControl"), patch.object(self.app, "connect_light"), \
                patch.object(self.app, "disconnect_light"), patch.object(self.app.demo, "launch"):
            self.app.launchApp(config_path=preprocess, use_camera=False, inbrowser=False)
            self.assertEqual(self.app.model_error, "model unavailable")
            self.assertEqual(self.app.roiThresholdConfig().path, self.path)
            self.main_callbacks["roi_threshold_save"](.47)
            self.assertEqual(self.store.load(), .47)

    def test_brightness_readouts_use_frontend_only_slider_change_events(self):
        from src.hardware.led_control import LIGHT_GROUPS

        labels = {f"{group.name} 밝기 (%)" for group in LIGHT_GROUPS}
        for demo in (self.app.demo, self.cam_demo):
            sliders = [block for block in demo.blocks.values() if getattr(block, "label", "") in labels]
            self.assertEqual(len(sliders), 4)
            for slider in sliders:
                listeners = [dependency for dependency in demo.config["dependencies"]
                             if any(tuple(target) == (slider._id, "change") for target in dependency["targets"])
                             and dependency.get("js")]
                self.assertEqual(len(listeners), 1)
                listener = listeners[0]
                self.assertFalse(listener["backend_fn"])
                self.assertEqual(listener["inputs"], [slider._id])
                readout = demo.blocks[listener["outputs"][0]]
                self.assertIn(f"{slider.value:g}%", readout.value)
                self.assertIn("설정 밝기", readout.value)


if __name__ == "__main__":
    unittest.main()
