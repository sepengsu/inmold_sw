"""In-memory camera preprocessing and ROI inspection behavior."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from time import monotonic

import numpy as np

from src.inspection.engine import InspectionEngine, annotate_frame, plan_measurement_regions, score_regions
from src.preprocess import preprocess_array, preprocess_roi_array


class GuiIntegrationTests(unittest.TestCase):
    def test_in_memory_preprocess_does_not_write_batch_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {
                "roi": {"x": 1, "y": 1, "width": 4, "height": 3},
                "clahe": {"enabled": False},
                "tiling": {"patch_size": 4, "stride": 2},
                "output": {"root_dir": folder},
            }
            result = preprocess_array(np.zeros((6, 7), dtype=np.uint8), config)
            self.assertEqual(result["roi_shape"], (3, 4))
            self.assertEqual(result["metadata"]["roi"], config["roi"])
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_three_scores_and_overlay_use_same_regions(self):
        exposures = tuple(np.zeros((512, 256), dtype=bool) for _ in range(3))
        exposures[1][256:] = True
        exposures[2][:] = True
        self.assertEqual(score_regions(exposures), (100.0, 50.0, 0.0))
        frame = np.full((2046, 2046, 3), 50, dtype=np.uint8)
        roi = {"x": 0, "y": 500, "width": 2046, "height": 800}
        regions = plan_measurement_regions(roi, np.ones((800, 2046), dtype=bool))
        overlay, crops = annotate_frame(frame, exposures, regions)
        self.assertEqual(overlay.shape, frame.shape)
        self.assertEqual([crop.shape for crop in crops], [(512, 256, 3)] * 3)
        for crop, region in zip(crops, regions):
            x, y = region["x"], region["y"]
            np.testing.assert_array_equal(crop, overlay[y:y + 512, x:x + 256])
        np.testing.assert_array_equal(overlay[1400, 1000], frame[1400, 1000])
        np.testing.assert_array_equal(overlay[700, 700], frame[700, 700])
        np.testing.assert_array_equal(overlay[800, 450], frame[800, 450])
        self.assertGreater(int(overlay[1000, 1000, 0]), 50)
        self.assertGreater(int(overlay[800, 1550, 0]), 50)

    def test_engine_infers_only_six_native_tiles_and_scores_the_point_pixels(self):
        engine = InspectionEngine.__new__(InspectionEngine)
        engine.config = {
            "roi": {"x": 10, "y": 100, "width": 2000, "height": 800},
            "clahe": {"enabled": False},
            "tiling": {"patch_size": 256, "stride": 128},
            "output": {"root_dir": "data", "save_roi": False,
                       "save_clahe": False, "save_tiles": False, "save_metadata": False},
        }
        engine.model = object()
        engine.batch_size = 2
        engine.checkpoint = {"threshold": 0.75}
        # Unmeasured pixels are all bright; they must never enter model inputs or scores.
        frame = np.full((1100, 2046, 3), 255, dtype=np.uint8)
        regions = plan_measurement_regions(engine.config["roi"], np.ones((800, 2000), dtype=bool))
        values = (0, 128, 255, 0, 128, 255)
        for index, region in enumerate(regions):
            x, y = region["x"], region["y"]
            for half in (0, 1):
                frame[y + half * 256:y + (half + 1) * 256, x:x + 256] = values[index * 2 + half]
        original = frame.copy()
        with patch("src.preprocess.pipeline.tile_image", side_effect=AssertionError("whole ROI tiled")), patch(
            "src.inspection.engine.predict_tiles", side_effect=lambda model, patches, **kwargs: [
            tile.astype(np.float32) / 255 for tile in patches
        ]) as predict:
            result = engine.inspect(frame)
        patches = predict.call_args.args[1]
        predict.assert_called_once()
        self.assertEqual(len(patches), 6)
        for tile, value in zip(patches, values):
            self.assertEqual(tile.shape, (256, 256))
            self.assertTrue(np.all(tile == value))
        self.assertEqual(predict.call_args.kwargs["batch_size"], 2)
        self.assertEqual(result.scores, (100.0, 50.0, 50.0))
        self.assertEqual(result.defect_pixels, (0, 65536, 65536))
        self.assertEqual(result.valid_pixels, (131072, 131072, 131072))
        for score, defects, total in zip(result.scores, result.defect_pixels, result.valid_pixels):
            self.assertEqual(score, (1 - defects / total) * 100)
        self.assertAlmostEqual(result.average, 200 / 3)
        self.assertEqual(result.regions, regions)
        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(result.annotated_frame[500, 700], frame[500, 700])

    def test_point_denominator_is_all_131072_pixels(self):
        exposures = tuple(np.zeros((512, 256), dtype=bool) for _ in range(3))
        exposures[0][0, 0] = True
        self.assertAlmostEqual(score_regions(exposures)[0], 100 * (1 - 1 / 131072))
        with self.assertRaises(ValueError):
            score_regions((np.zeros((256, 256), dtype=bool),) * 3)

    def test_points_keep_fixed_spacing_and_roi_vertical_midpoint(self):
        roi = {"x": 0, "y": 400, "width": 2046, "height": 1000}
        regions = plan_measurement_regions(roi, np.ones((1000, 2046), dtype=bool))
        self.assertEqual([r["x"] for r in regions], [348, 895, 1442])
        self.assertEqual([r["y"] for r in regions], [644] * 3)
        for region in regions:
            self.assertEqual((region["width"], region["height"]), (256, 512))
        roi["height"] = 1001
        odd = plan_measurement_regions(roi, np.ones((1001, 2046), dtype=bool))
        self.assertEqual([r["y"] for r in odd], [645] * 3)

    def test_too_small_or_partly_outside_points_stop_before_inference(self):
        for width, height in ((1349, 800), (2046, 511)):
            with self.subTest(width=width, height=height), self.assertRaisesRegex(ValueError, "크기가 부족"):
                plan_measurement_regions({"x": 0, "y": 0, "width": width, "height": height},
                                         np.ones((height, width), dtype=bool))
        engine = InspectionEngine.__new__(InspectionEngine)
        engine.config = {"roi": {"x": 0, "y": 0, "width": 1200, "height": 800},
                         "clahe": {"enabled": False}}
        with patch("src.inspection.engine.predict_tiles") as predict:
            with self.assertRaisesRegex(ValueError, "크기가 부족"):
                engine.inspect(np.full((800, 1200, 3), 50, dtype=np.uint8))
        predict.assert_not_called()
        roi = {"x": 0, "y": 0, "width": 2046, "height": 800}
        valid = np.ones((800, 2046), dtype=bool)
        valid[144, 348] = False
        with self.assertRaisesRegex(ValueError, "P1.*경계"):
            plan_measurement_regions(roi, valid)

    def test_roi_only_preprocessing_preserves_existing_clahe_without_full_tiling(self):
        image = np.random.default_rng(42).integers(0, 256, size=(600, 1500), dtype=np.uint8)
        config = {"roi": {"x": 10, "y": 20, "width": 1400, "height": 550},
                  "clahe": {"enabled": True}}
        previous = preprocess_array(image, config)
        with patch("src.preprocess.pipeline.tile_image", side_effect=AssertionError("whole ROI tiled")):
            current = preprocess_roi_array(image, config)
        for key in ("roi", "clahe", "valid_mask"):
            np.testing.assert_array_equal(current[key], previous[key])
        self.assertEqual(current["metadata"]["roi"], previous["metadata"]["roi"])


class CameraFrameTests(unittest.TestCase):
    def test_disconnected_and_stale_frames_are_not_returned(self):
        from src.hardware.camera_control import CameraControl

        camera = CameraControl()
        camera._frame = SimpleNamespace(display_rgb=np.zeros((4, 6, 3), dtype=np.uint8))
        camera._last_frame_at = monotonic()
        self.assertIsNone(camera.get_latest_frame())
        camera._connected = True
        frame = camera.get_latest_frame()
        frame[:] = 255
        self.assertFalse(camera.get_latest_frame().any())
        camera._last_frame_at = monotonic() - 3
        self.assertIsNone(camera.get_latest_frame())

    def test_closing_camera_clears_previous_frame(self):
        from src.hardware.camera_control import CameraControl

        camera = CameraControl()
        camera._frame = object()
        camera._connected = True
        camera.stop()
        self.assertIsNone(camera._frame)
        self.assertFalse(camera.get_status().connected)
