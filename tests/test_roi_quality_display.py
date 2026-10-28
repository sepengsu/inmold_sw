"""Distinguish a passing score from other ROI rejection gates in both UIs."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from src.preprocess.auto_roi import AutoRoiResult, RoiQualityError, detect_auto_roi
from src.roi import roi_boundary_quality as quality
from src.roi.quality_display import roi_quality_message, roi_score_message
from src.roi.threshold_config import RoiThresholdConfigStore
from src.roi.quality_profiles import ROI_QUALITY_PROFILES


def reviewed_result() -> AutoRoiResult:
    config = quality.load_config()
    for section, values in ROI_QUALITY_PROFILES["medium"].items():
        config[section].update(deepcopy(values))
    config["quality"]["min_score"] = .30
    fit = {"valid": True, "inlier_ratio": .8, "x_coverage": .8, "max_gap_ratio": .1,
           "rmse": 4., "median_residual_px": 3., "left_support": 4, "center_support": 4, "right_support": 4}
    candidate = {"roi_quality": .34, "top_line": {"slope": 0., "intercept": 100.},
                 "bottom_line": {"slope": 0., "intercept": 300.},
                 "top_fit": {**fit, "left_support": 1},
                 "bottom_fit": {**fit, "valid": False, "inlier_ratio": .35}}
    assessment = quality.assess(candidate, [candidate], (512, 960), config,
                                {"top_unavailable": False, "bottom_unavailable": False}, 0, 1)
    metadata = {**assessment, "roi_detected": True, "quality_score": .34,
                "quality_threshold": .30, "quality_limits": config["quality"],
                "boundary_fits": {"top": candidate["top_fit"], "bottom": candidate["bottom_fit"]}}
    return AutoRoiResult(np.ones((3, 4), bool), metadata)


class RoiQualityDisplayTests(unittest.TestCase):
    def test_point_34_passes_point_30_but_geometric_failures_still_block(self):
        result = reviewed_result()
        self.assertFalse(result.accepted)
        self.assertNotIn("LOW_BOUNDARY_SCORE", result.metadata["needs_review_reason"])
        self.assertEqual(set(result.metadata["needs_review_reason"].split(";")),
                         {"MISSING_THIRD_SUPPORT_TOP", "RANSAC_FAILED_BOTTOM", "LOW_INLIER_RATIO_BOTTOM"})
        message = roi_quality_message(result.metadata)
        self.assertIn("점수 조건 통과: 0.340 ≥ 적용 기준 0.300", message)
        self.assertIn("상단 지지점 부족: 왼쪽 1개 / 각 구간 최소 2개", message)
        self.assertIn("하단 경계 일치 비율 35.0% < 최소 55.0%", message)
        self.assertIn("하단 경계 직선 피팅 실패", message)

    def test_score_failure_is_explicit(self):
        metadata = deepcopy(reviewed_result().metadata)
        metadata.update(quality_score=.29, needs_review_reason="LOW_BOUNDARY_SCORE")
        self.assertIn("점수 조건 미달: 0.290 < 적용 기준 0.300", roi_quality_message(metadata))

    def test_normal_inference_warning_does_not_claim_inspection_is_blocked(self):
        for reason, label in (("FABRIC_OUTSIDE_TOP", "상단 ROI 바깥에도 직물 무늬가 검출됨"),
                              ("AMBIGUOUS_CANDIDATES", "서로 다른 경계 후보들의 점수가 비슷해 선택이 불확실함")):
            with self.subTest(reason=reason):
                metadata = deepcopy(reviewed_result().metadata)
                metadata.update(inspection_allowed=True, needs_review_reason=reason, warning_reasons=[reason])
                message = roi_quality_message(metadata)
                self.assertIn("ROI 경고 — 점수 통과 · 추론 허용:", message)
                self.assertIn(f"{label} [{reason}]", message)
                self.assertNotIn("자동 검사 보류 사유", message)
                metadata.pop("inspection_allowed")
                self.assertIn("자동 검사 보류 사유", roi_quality_message(metadata))

    def test_passing_result_has_no_hold_reasons(self):
        metadata = deepcopy(reviewed_result().metadata)
        metadata.update(quality_status="RECOVERED", needs_review_reason="")
        self.assertTrue(AutoRoiResult(np.ones((3, 4), bool), metadata).accepted)
        self.assertEqual(roi_quality_message(metadata), "점수 조건 통과: 0.340 ≥ 적용 기준 0.300")

    def test_extraction_keeps_threshold_snapshot_after_config_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roi.yaml"
            config = quality.load_config()
            config["quality"]["min_score"] = .30
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = detect_auto_roi(np.full((256, 320), 120, np.uint8), path, require_accepted=False)
            RoiThresholdConfigStore(path).save(.8)
            self.assertEqual(result.metadata["quality_threshold"], .30)
            self.assertIn("기준 0.300", roi_score_message(result.metadata))
            self.assertEqual(result.metadata["quality_limits"]["min_score"], .30)
            self.assertIn("boundary_fits", json.loads(json.dumps(result.metadata)))
