"""Persist shared lighting values without transmitting them to the controller."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

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
