"""Native ROI geometry, quality rejection, and valid-pixel inspection scores."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.inspection.engine import InspectionEngine
from src.postprocess import impregnation_pct, merge_probabilities, restore_original_mask
from src.preprocess import load_config, preprocess_array, valid_roi_mask
from src.preprocess.auto_roi import AutoRoiResult, RoiQualityError, detect_auto_roi


def tilted_roi() -> AutoRoiResult:
    top = np.array([1, 1, 2, 2, 3, 3])
    bottom = np.array([3, 4, 4, 5, 5, 6])
    rows = np.arange(6)[:, None]
    return AutoRoiResult((rows >= top) & (rows < bottom), {
        "method": "gabor_boundary_quality_v5", "source_shape": [6, 6],
        "roi_detected": True, "quality_status": "PASS", "quality_score": .8,
        "needs_review_reason": "", "top_y": top.tolist(), "bottom_y_exclusive": bottom.tolist(),
    })


def auto_config() -> dict:
    return {"roi": {"x": 0, "y": 0, "width": 2000, "height": 1200},
            "auto_roi": {"enabled": True}, "clahe": {"enabled": False},
            "tiling": {"patch_size": 4, "stride": 2}}


def large_tilted_roi() -> AutoRoiResult:
    top = 400 + np.floor(.04 * np.arange(2046)).astype(int)
    bottom = top + 800
    rows = np.arange(2046)[:, None]
    return AutoRoiResult((rows >= top) & (rows < bottom), {
        "method": "gabor_boundary_quality_v5", "source_shape": [2046, 2046],
        "roi_detected": True, "quality_status": "PASS", "quality_score": .8,
        "needs_review_reason": "", "top_y": top.tolist(), "bottom_y_exclusive": bottom.tolist(),
    })


class RoiPipelineTests(unittest.TestCase):
    def test_native_mask_preserves_pixels_and_roundtrips_metadata(self):
        detected = tilted_roi()
        image = np.arange(36, dtype=np.uint8).reshape(6, 6) + 1
        original = image.copy()
        with patch("src.preprocess.auto_roi.detect_auto_roi", return_value=detected):
            result = preprocess_array(image, auto_config())
        self.assertEqual(result["original_shape"], (6, 6))
        self.assertEqual(result["roi_shape"], (5, 6))
        self.assertEqual(result["metadata"]["roi"], {"x": 0, "y": 1, "width": 6, "height": 5})
        valid = result["valid_mask"]
        np.testing.assert_array_equal(result["roi"][valid], original[1:6][valid])
        self.assertFalse(result["roi"][~valid].any())
        np.testing.assert_array_equal(image, original)
        metadata = json.loads(json.dumps(result["metadata"]))
        np.testing.assert_array_equal(valid_roi_mask(metadata), valid)
        predictions = [np.ones((4, 4), dtype=np.float32) for _ in result["patches"]]
        merged = merge_probabilities(predictions, metadata)
        np.testing.assert_array_equal(restore_original_mask(merged > .5, metadata), detected.mask)

    def test_background_is_excluded_from_roi_denominator(self):
        valid = np.array([[True, False, True, True, False, True],
                          [True, True, False, True, True, False]])
        exposure = np.ones_like(valid)
        exposure[0, 0] = False
        exposure[0, 2] = False
        exposure[0, 3] = False
        exposure[1, 4] = False
        self.assertAlmostEqual(impregnation_pct(exposure, valid), 50)
        with self.assertRaises(ValueError):
            impregnation_pct(exposure, np.zeros_like(valid))

    def test_bad_geometry_cannot_clip_or_resize_the_roi(self):
        with patch("src.preprocess.auto_roi.detect_auto_roi", return_value=tilted_roi()):
            result = preprocess_array(np.full((6, 6), 50, np.uint8), auto_config())
        metadata = result["metadata"]
        metadata["roi_detection"]["top_y"][0] = .5
        with self.assertRaises(ValueError):
            valid_roi_mask(metadata)

    def test_inspection_keeps_tilted_roi_and_places_six_tiles_on_its_midline(self):
        detected = large_tilted_roi()
        engine = InspectionEngine.__new__(InspectionEngine)
        engine.config, engine.model, engine.batch_size = auto_config(), object(), 2
        engine.checkpoint = {"threshold": .5}
        with patch("src.preprocess.auto_roi.detect_auto_roi", return_value=detected), patch(
            "src.inspection.engine.predict_tiles",
            side_effect=lambda model, patches, **kwargs: [np.ones(tile.shape, np.float32) for tile in patches],
        ) as predict:
            result = engine.inspect(np.full((2046, 2046, 3), 50, np.uint8))
        self.assertEqual(len(predict.call_args.args[1]), 6)
        self.assertEqual(result.scores, (0, 0, 0))
        self.assertEqual(result.average, 0)
        self.assertEqual(result.roi_detection["quality_status"], "PASS")
        self.assertEqual(result.annotated_frame.shape, (2046, 2046, 3))
        self.assertIs(result.roi_detection, detected.metadata)
        self.assertLess(result.regions[0]["y"], result.regions[1]["y"])
        self.assertLess(result.regions[1]["y"], result.regions[2]["y"])
        for region in result.regions:
            x, y = region["x"], region["y"]
            self.assertTrue(detected.mask[y:y + 512, x:x + 256].all())
            column = x + 128
            roi_midpoint = (detected.metadata["top_y"][column]
                            + detected.metadata["bottom_y_exclusive"][column] - 1) / 2
            self.assertLessEqual(abs(y + 255.5 - roi_midpoint), .5)
        np.testing.assert_array_equal(result.annotated_frame[900, 700], [50, 50, 50])

    def test_partly_invalid_point_is_rejected_before_model(self):
        detected = large_tilted_roi()
        # A narrow left band cannot hold a full-height P1 even though the ROI
        # bounding box is large enough. Do not move or pad that point.
        detected.mask[:850, 348:604] = False
        detected.metadata["top_y"][348:604] = [850] * 256
        engine = InspectionEngine.__new__(InspectionEngine)
        engine.config = auto_config()
        with patch("src.preprocess.auto_roi.detect_auto_roi", return_value=detected), patch(
            "src.inspection.engine.predict_tiles"
        ) as predict:
            with self.assertRaisesRegex(ValueError, "P1.*경계"):
                engine.inspect(np.full((2046, 2046, 3), 50, np.uint8))
        predict.assert_not_called()

    def test_review_result_never_reaches_model_inference(self):
        detected = tilted_roi()
        detected.metadata.update(quality_status="NEEDS_REVIEW", needs_review_reason="LOW_BOUNDARY_SCORE")
        engine = InspectionEngine.__new__(InspectionEngine)
        engine.config = auto_config()
        with patch("src.preprocess.auto_roi.detect_auto_roi", side_effect=RoiQualityError(detected)), patch(
            "src.inspection.engine.predict_tiles"
        ) as predict:
            with self.assertRaisesRegex(RoiQualityError, "검토 필요"):
                engine.inspect(np.full((6, 6, 3), 50, np.uint8))
        predict.assert_not_called()
        self.assertEqual(engine.last_roi_detection["quality_status"], "NEEDS_REVIEW")

    def test_score_passed_quality_warnings_infer_exactly_six_native_tiles(self):
        for reason in ("FABRIC_OUTSIDE_TOP", "FABRIC_OUTSIDE_BOTTOM",
                       "FABRIC_OUTSIDE_TOP;FABRIC_OUTSIDE_BOTTOM", "AMBIGUOUS_CANDIDATES",
                       "AMBIGUOUS_CANDIDATES;FABRIC_OUTSIDE_TOP",
                       "LOW_INLIER_RATIO_BOTTOM", "MISSING_THIRD_SUPPORT_TOP",
                       "RANSAC_FAILED_TOP", "LOW_X_COVERAGE_BOTTOM", "LONG_UNSUPPORTED_GAP_TOP",
                       "HIGH_RANSAC_RMSE_BOTTOM", "HIGH_MEDIAN_RESIDUAL_TOP",
                       "OUTSIDE_CHECK_UNAVAILABLE_BOTTOM", "IMPLAUSIBLE_HEIGHT",
                       "EXCESSIVE_TILT", "LARGE_ANGLE_DIFFERENCE", "LARGE_HEIGHT_VARIATION",
                       "LOW_MULTI_EXPOSURE_CONSENSUS"):
            with self.subTest(reason=reason):
                detected = large_tilted_roi()
                detected.metadata.update(quality_status="NEEDS_REVIEW", quality_score=.44,
                                         quality_threshold=.44, needs_review_reason=reason)
                engine = InspectionEngine.__new__(InspectionEngine)
                engine.config, engine.model, engine.batch_size = auto_config(), object(), 2
                engine.checkpoint = {"threshold": .5}
                with patch("src.preprocess.auto_roi.detect_auto_roi", side_effect=RoiQualityError(detected)), patch(
                    "src.inspection.engine.predict_tiles", side_effect=lambda model, patches, **kwargs: [
                        np.zeros(tile.shape, np.float32) for tile in patches
                    ],
                ) as predict:
                    result = engine.inspect(np.full((2046, 2046, 3), 50, np.uint8))
                patches = predict.call_args.args[1]
                self.assertEqual([tile.shape for tile in patches], [(256, 256)] * 6)
                self.assertEqual(result.scores, (100., 100., 100.))
                self.assertEqual(result.valid_pixels, (131072,) * 3)
                self.assertEqual(result.roi_detection["quality_status"], "NEEDS_REVIEW")
                self.assertTrue(result.roi_detection["inspection_allowed"])
                self.assertEqual(result.roi_detection["warning_reasons"], reason.split(";"))
                self.assertNotIn("inspection_allowed", detected.metadata)
                for region in result.regions:
                    x, y = region["x"], region["y"]
                    self.assertTrue(detected.mask[y:y + 512, x:x + 256].all())

    def test_ambiguous_candidates_point_413_passes_point_250_and_preserves_real_predictions(self):
        detected = large_tilted_roi()
        detected.metadata.update(quality_status="NEEDS_REVIEW", quality_score=.413,
                                 quality_threshold=.250, needs_review_reason="AMBIGUOUS_CANDIDATES")
        engine = InspectionEngine.__new__(InspectionEngine)
        engine.config, engine.model, engine.batch_size = auto_config(), object(), 8
        engine.checkpoint = {"threshold": .5}
        outputs = [np.full((256, 256), value, np.float32) for value in (0, 0, 0, 1, 1, 1)]
        with patch("src.preprocess.auto_roi.detect_auto_roi", side_effect=RoiQualityError(detected)), patch(
            "src.inspection.engine.predict_tiles", return_value=outputs,
        ) as predict:
            result = engine.inspect(np.full((2046, 2046, 3), 50, np.uint8))
        predict.assert_called_once()
        self.assertEqual(len(predict.call_args.args[1]), 6)
        self.assertEqual(result.scores, (100., 50., 0.))
        self.assertTrue(result.roi_detection["inspection_allowed"])
        self.assertEqual(result.roi_detection["warning_reasons"], ["AMBIGUOUS_CANDIDATES"])

    def test_warning_never_overrides_low_score_missing_roi_or_invalid_geometry(self):
        cases = (
            {"quality_score": .089}, {"quality_score": float("nan")},
            {"quality_threshold": float("nan")}, {"roi_detected": False},
            {"needs_review_reason": "AMBIGUOUS_CANDIDATES", "quality_score": .249, "quality_threshold": .250},
            {"needs_review_reason": "AMBIGUOUS_CANDIDATES;LOW_BOUNDARY_SCORE"},
            {"needs_review_reason": "FABRIC_OUTSIDE_TOP;INVALID_GEOMETRY"},
            {"quality_status": "FAIL"},
            {"needs_review_reason": ""},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                detected = large_tilted_roi()
                detected.metadata.update(quality_status="NEEDS_REVIEW", quality_score=.44,
                                         quality_threshold=.09, needs_review_reason="FABRIC_OUTSIDE_TOP")
                detected.metadata.update(changes)
                engine = InspectionEngine.__new__(InspectionEngine)
                engine.config = auto_config()
                with patch("src.preprocess.auto_roi.detect_auto_roi", side_effect=RoiQualityError(detected)), patch(
                    "src.inspection.engine.predict_tiles"
                ) as predict:
                    with self.assertRaises(RoiQualityError):
                        engine.inspect(np.full((2046, 2046, 3), 50, np.uint8))
                predict.assert_not_called()

    def test_quality_warning_does_not_override_point_geometry_or_default_preprocessing(self):
        detected = large_tilted_roi()
        detected.metadata.update(quality_status="NEEDS_REVIEW", quality_score=.44,
                                 quality_threshold=.09, needs_review_reason="AMBIGUOUS_CANDIDATES")
        engine = InspectionEngine.__new__(InspectionEngine)
        engine.config = auto_config()
        frame = np.full((2046, 2046, 3), 50, np.uint8)
        with patch("src.preprocess.auto_roi.detect_auto_roi", side_effect=RoiQualityError(detected)):
            with self.assertRaises(RoiQualityError):
                preprocess_array(frame[:, :, 0], auto_config())
        detected.mask[:850, 348:604] = False
        detected.metadata["top_y"][348:604] = [850] * 256
        with patch("src.preprocess.auto_roi.detect_auto_roi", side_effect=RoiQualityError(detected)), patch(
            "src.inspection.engine.predict_tiles"
        ) as predict:
            with self.assertRaisesRegex(ValueError, "P1.*경계"):
                engine.inspect(frame)
        predict.assert_not_called()

    def test_boundary_config_path_is_relative_to_yaml(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "preprocess.yaml"
            path.write_text("roi: {x: 0, y: 0, width: 6, height: 6}\nauto_roi: {enabled: true, boundary_config: boundary.yaml}\n", encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config["auto_roi"]["boundary_config"], Path(folder).resolve() / "boundary.yaml")


class CertificationRoiScoreTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((180, 320, 3), 240, np.uint8)
        for x in (30, 130, 230):
            self.frame[80:110, x:x + 30] = 100
        self.engine = InspectionEngine.__new__(InspectionEngine)
        self.engine.config, self.engine.model = auto_config(), object()
        self.engine.checkpoint = {"threshold": .5}

    def detection(self, *, score=.413, threshold=.25, reason=""):
        height, width = self.frame.shape[:2]
        rows = np.arange(height)[:, None]
        return AutoRoiResult(np.broadcast_to((rows >= 40) & (rows < 160), (height, width)).copy(), {
            "source_shape": [height, width], "roi_detected": True,
            "quality_status": "NEEDS_REVIEW" if reason else "PASS",
            "quality_score": score, "quality_threshold": threshold, "needs_review_reason": reason,
            "top_y": [40] * width, "bottom_y_exclusive": [160] * width,
        })

    def test_common_score_pass_preserves_certification_three_window_inference(self):
        for score, reason in ((.413, ""), (.413, "AMBIGUOUS_CANDIDATES"), (.25, "FABRIC_OUTSIDE_TOP")):
            with self.subTest(score=score, reason=reason):
                detected = self.detection(score=score, reason=reason)
                detector = (patch("src.preprocess.auto_roi.detect_auto_roi", side_effect=RoiQualityError(detected))
                            if reason else patch("src.preprocess.auto_roi.detect_auto_roi", return_value=detected))
                predictions = [[np.full((256, 256), value, np.float32)] for value in (0, 1, 0)]
                original = self.frame.copy()
                with detector, patch("src.PCE.inspection.predict_tiles", side_effect=predictions) as predict:
                    result = self.engine.inspect_certification(self.frame)
                self.assertEqual(predict.call_count, 3)
                for call in predict.call_args_list:
                    self.assertIs(call.args[0], self.engine.model)
                    self.assertEqual([tile.shape for tile in call.args[1]], [(256, 256)])
                    self.assertEqual(call.kwargs["batch_size"], 1)
                self.assertEqual(result.scores, (100., 0., 100.))
                self.assertEqual(result.valid_pixels, (24 * 24,) * 3)
                self.assertEqual([r["x"] for r in result.regions], [30, 130, 230])
                self.assertEqual([r["quad"][0][1] for r in result.regions], [83.] * 3)
                self.assertEqual(result.annotated_frame.shape, self.frame.shape)
                self.assertEqual(result.roi_detection["quality_score"], score)
                self.assertEqual(result.roi_detection["quality_threshold"], .25)
                self.assertEqual(result.roi_detection, self.engine.last_roi_detection)
                self.assertEqual(result.pce_detection["roi_inset_px"], 3)
                if reason:
                    self.assertTrue(result.roi_detection["inspection_allowed"])
                    self.assertEqual(result.roi_detection["warning_reasons"], [reason])
                np.testing.assert_array_equal(self.frame, original)

    def test_common_score_failure_or_invalid_roi_blocks_before_certification_detection(self):
        for changes in ({"quality_score": .249}, {"quality_score": float("nan")},
                        {"roi_detected": False}, {"needs_review_reason": "INVALID_GEOMETRY"}):
            with self.subTest(changes=changes):
                detected = self.detection(reason="AMBIGUOUS_CANDIDATES")
                detected.metadata.update(changes)
                with patch("src.preprocess.auto_roi.detect_auto_roi", side_effect=RoiQualityError(detected)), patch(
                    "src.PCE.inspect_frame"
                ) as certification:
                    with self.assertRaises(RoiQualityError):
                        self.engine.inspect_certification(self.frame)
                certification.assert_not_called()
                self.assertEqual(self.engine.last_roi_detection, detected.metadata)

    def test_certification_still_requires_three_valid_windows_after_common_score_pass(self):
        self.frame[80:110, 230:260] = 240
        with patch("src.preprocess.auto_roi.detect_auto_roi", return_value=self.detection()), patch(
            "src.PCE.inspection.predict_tiles"
        ) as predict:
            with self.assertRaisesRegex(ValueError, "종이 내부.*2개"):
                self.engine.inspect_certification(self.frame)
        predict.assert_not_called()


class UpstreamDetectorTests(unittest.TestCase):
    def test_real_gabor_detector_finds_tilted_synthetic_band(self):
        # Same textured-band construction used by the upstream boundary tests.
        ys, xs = np.mgrid[:512, :960]
        inside = (ys >= 150 + .015 * xs) & (ys <= 350 - .005 * xs)
        texture = 90 + 45 * np.sin(2 * np.pi * xs / 16) * np.sin(2 * np.pi * ys / 16)
        image = np.where(inside, texture, 45).astype(np.uint8)
        result = detect_auto_roi(image, require_accepted=False)
        self.assertEqual(result.mask.shape, image.shape)
        self.assertTrue(result.metadata["roi_detected"])
        self.assertLess(abs(result.metadata["top_line"]["intercept"] - 150), 25)
        self.assertLess(abs(result.metadata["bottom_line"]["intercept"] - 350), 25)

    def test_blank_frame_is_reviewed_instead_of_full_frame_acceptance(self):
        with self.assertRaises(RoiQualityError) as caught:
            detect_auto_roi(np.full((256, 320), 120, np.uint8))
        self.assertFalse(caught.exception.result.accepted)
        self.assertFalse(caught.exception.result.metadata["roi_detected"])


if __name__ == "__main__":
    unittest.main()
