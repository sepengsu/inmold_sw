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
