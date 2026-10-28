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

    def test_boundary_config_path_is_relative_to_yaml(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "preprocess.yaml"
            path.write_text("roi: {x: 0, y: 0, width: 6, height: 6}\nauto_roi: {enabled: true, boundary_config: boundary.yaml}\n", encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config["auto_roi"]["boundary_config"], Path(folder).resolve() / "boundary.yaml")


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
