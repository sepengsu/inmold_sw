"""Linked quality gates and configurable support during actual line fitting."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from src.preprocess.auto_roi import RoiQualityError, detect_auto_roi
from src.roi import auto_roi_local_std as core
from src.roi import roi_boundary as boundary
from src.roi.roi_boundary_quality import load_config
from src.roi.quality_profiles import ROI_QUALITY_PROFILES, linked_roi_settings, roi_level
from src.roi.threshold_config import RoiThresholdConfigStore


class RoiQualityProfilesTests(unittest.TestCase):
    def test_profile_anchors_and_intermediate_score_adjust_all_linked_gates(self):
        for name, profile in ROI_QUALITY_PROFILES.items():
            config = linked_roi_settings(profile["quality"]["min_score"])
            self.assertEqual(config, profile)
            self.assertEqual(roi_level(config), name)
        between = linked_roi_settings(.325)
        self.assertIsNone(roi_level(between))
        self.assertAlmostEqual(between["quality"]["min_inlier_ratio"], .45)
        self.assertAlmostEqual(between["quality"]["min_x_coverage"], .55)
        self.assertAlmostEqual(between["quality"]["max_gap_ratio"], .425)
        self.assertEqual(between["quality"]["min_third_support"], 1)
        self.assertEqual(between["ransac"]["thresholds"], [10., 15., 20.])

    def test_increasing_score_never_relaxes_any_linked_gate(self):
        increasing = ("min_inlier_ratio", "min_x_coverage", "min_third_support",
                      "ambiguity_score_gap", "min_consensus_ratio")
        decreasing = ("max_gap_ratio", "max_rmse_px", "max_median_residual_px", "max_tilt_deg",
                      "max_angle_diff_deg", "max_height_variation_px", "ambiguity_boundary_gap_px")
        previous = linked_roi_settings(0.)
        for score in np.linspace(0, 1, 101):
            current = linked_roi_settings(float(score))
            for key in increasing:
                self.assertGreaterEqual(current["quality"][key], previous["quality"][key])
            for key in decreasing:
                self.assertLessEqual(current["quality"][key], previous["quality"][key])
            self.assertTrue(all(a <= b for a, b in zip(current["ransac"]["thresholds"], previous["ransac"]["thresholds"])))
            previous = current

    def test_ransac_can_follow_zero_third_support_without_bypassing_actual_fit(self):
        points = [(x, 100 + .01 * x) for x in (400, 500, 600, 700, 800, 900)]
        arguments = dict(min_span=400, residual_threshold=12, x_bounds=(0, 1000),
                         max_rmse=14, min_inlier_ratio=.35, max_gap_ratio=.60, max_median_residual=14)
        low = core.fit_boundary_line(points, **arguments, min_third_support=0)
        legacy = core.fit_boundary_line(points, **arguments)
        self.assertTrue(low["valid"])
        self.assertEqual(low["left_support"], 0)
        self.assertFalse(legacy["valid"])
        self.assertEqual(legacy["rejection_reason"], "missing_third_support")
        self.assertFalse(core.fit_boundary_line(points[:3], **arguments, min_third_support=0)["valid"])

    def test_detector_wrapper_uses_selected_profile_in_line_fitting(self):
        points = [(x, 100 + .01 * x) for x in (400, 500, 600, 700, 800, 900)]
        results = {}
        for name in ("low", "medium"):
            config = load_config()
            for section, values in ROI_QUALITY_PROFILES[name].items():
                config[section].update(deepcopy(values))
            results[name] = boundary.fit_line(points, 1000, config)
        self.assertTrue(results["low"]["valid"])
        self.assertFalse(results["medium"]["valid"])

    def test_blank_frame_never_becomes_accepted_in_any_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roi.yaml"
            path.write_text(yaml.safe_dump(load_config()), encoding="utf-8")
            store = RoiThresholdConfigStore(path)
            for name, profile in ROI_QUALITY_PROFILES.items():
                with self.subTest(name=name):
                    store.save(profile["quality"]["min_score"])
                    with self.assertRaises(RoiQualityError) as caught:
                        detect_auto_roi(np.full((256, 320), 120, np.uint8), path)
                    self.assertFalse(caught.exception.result.accepted)
                    self.assertFalse(caught.exception.result.metadata["roi_detected"])
                    self.assertEqual(caught.exception.result.metadata["validation_level"], name)


if __name__ == "__main__":
    unittest.main()
