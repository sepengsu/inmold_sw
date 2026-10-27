"""Certification connected-component selection and one-at-a-time inference."""

import unittest
import tempfile
from unittest.mock import patch

import gradio as gr
import cv2
import numpy as np

from src.PCE.inspection import detect_regions, inspect_frame
from src.gui.statistics import StatisticsController
from src.inspection_history import InspectionHistoryStore, InspectionRecord


def sample_frame(points=3):
    frame = np.full((180, 320, 3), 240, dtype=np.uint8)
    for x in (30, 130, 230)[:points]:
        frame[80:110, x:x + 30] = 100
    return frame


class PceTests(unittest.TestCase):
    def test_detects_three_left_to_right_without_outer_background(self):
        frame = sample_frame()
        gray = frame[:, :, 0]
        regions = detect_regions(gray)
        self.assertEqual([item[0]["x"] for item in regions], [30, 130, 230])
        self.assertEqual([item[0]["area"] for item in regions], [900] * 3)
        self.assertTrue(all(item[1].all() for item in regions))

    def test_detects_full_resolution_pce_squares(self):
        gray = np.full((2046, 2046), 240, dtype=np.uint8)
        gray[:400] = 0  # The dark image border must not become a measurement.
        for x in (480, 980, 1480):
            gray[980:1095, x:x + 115] = 100
        regions = detect_regions(gray)
        self.assertEqual([item[0]["area"] for item in regions], [115 * 115] * 3)

    def test_rotated_squares_are_straightened_before_inference(self):
        frame = np.full((250, 550, 3), 240, dtype=np.uint8)
        for center_x in (100, 275, 450):
            corners = cv2.boxPoints(((center_x, 125), (76, 76), 22))
            cv2.fillConvexPoly(frame, np.rint(corners).astype(np.int32), (100, 100, 100))
        seen = []

        def infer(model, patches, *, batch_size):
            seen.append(patches[0].copy())
            return [np.zeros((256, 256), np.float32)]

        with patch("src.PCE.inspection.predict_tiles", side_effect=infer):
            result = inspect_frame(frame, object(), 0.5, {})
        self.assertEqual(len(seen), 3)
        for region, tile, preview in zip(result.regions, seen, result.crops, strict=True):
            x, y, w, h = (region[key] for key in ("x", "y", "width", "height"))
            self.assertAlmostEqual(region["rotation_deg"], 22, delta=2)
            self.assertLess(np.count_nonzero(tile > 150) / tile.size, 0.05)
            self.assertGreater(np.count_nonzero(frame[y:y + h, x:x + w, 0] > 150) / (w * h), 0.3)
            self.assertEqual(preview.shape, (256, 256, 3))
            self.assertEqual(result.annotated_frame[y, x].tolist(), [0, 0, 0])
        self.assertEqual(result.scores, (100.0, 100.0, 100.0))

        with tempfile.TemporaryDirectory() as folder:
            store = InspectionHistoryStore(folder)
            record = InspectionRecord(*result.scores, measured_at="2026-10-27T12:00:00+09:00",
                                      source="file", metadata={"regions": result.regions,
                                      "conditions": {"inspection_mode": "certification"}})
            saved = store.save(record, frame, result.annotated_frame)
            event = gr.SelectData(None, {"index": [0, 0], "value": None, "selected": True})
            detail = StatisticsController(lambda: store).detail([saved.record_id], event)
            self.assertEqual(detail[3].shape, (256, 256, 3))
            self.assertEqual(detail[3][128, 128].tolist(), [100, 100, 100])

    def test_rotated_black_pixels_stay_out_of_score_and_overlay(self):
        frame = np.full((250, 550, 3), 240, dtype=np.uint8)
        for center_x in (100, 275, 450):
            corners = cv2.boxPoints(((center_x, 125), (76, 76), 22))
            cv2.fillConvexPoly(frame, np.rint(corners).astype(np.int32), (100, 100, 100))
        frame[120:130, 95:105] = 0
        with patch("src.PCE.inspection.predict_tiles",
                   return_value=[np.ones((256, 256), np.float32)]):
            result = inspect_frame(frame, object(), 0.5, {})
        self.assertEqual(result.scores, (0.0, 0.0, 0.0))
        self.assertLess(result.valid_pixels[0], result.regions[0]["rectified_width"]
                        * result.regions[0]["rectified_height"])
        np.testing.assert_array_equal(result.annotated_frame[125, 100], [0, 0, 0])

    def test_fills_tiles_with_entire_crop_and_infers_one_at_a_time(self):
        frame = sample_frame()
        original = frame.copy()
        seen = []

        def infer(model, patches, *, batch_size):
            self.assertEqual(batch_size, 1)
            self.assertEqual(len(patches), 1)
            tile = patches[0]
            self.assertEqual(tile.shape, (256, 256))
            self.assertTrue(np.all(tile == 100))
            seen.append(tile.copy())
            probability = np.zeros((256, 256), np.float32)
            if len(seen) == 1:
                probability[:10, :10] = 1
            elif len(seen) == 2:
                probability[:] = 1
            return [probability]

        with patch("src.PCE.inspection.predict_tiles", side_effect=infer) as predict:
            result = inspect_frame(frame, object(), 0.5, {"clahe": {"enabled": False}})
        self.assertEqual(predict.call_count, 3)
        first_exposure = np.zeros((256, 256), np.uint8)
        first_exposure[:10, :10] = 1
        source_exposure = cv2.resize(first_exposure, (30, 30), interpolation=cv2.INTER_NEAREST)
        self.assertEqual(result.scores, (100 * (1 - np.count_nonzero(source_exposure) / 900),
                                         0.0, 100.0))
        self.assertEqual(result.valid_pixels, (900, 900, 900))
        self.assertEqual([region["x"] for region in result.regions], [30, 130, 230])
        self.assertEqual(result.crops[0].shape, (256, 256, 3))
        self.assertTrue(np.all(result.crops[0][-1, -1] == 100))
        self.assertTrue(np.all(result.annotated_frame[0:79] == 0))
        self.assertTrue(np.all(result.annotated_frame[80:110, 0:30] == 0))
        self.assertGreater(result.annotated_frame[80, 30, 0], 100)
        np.testing.assert_array_equal(frame, original)

    def test_crop_includes_bright_pixels_inside_its_bounding_box(self):
        frame = sample_frame()
        frame[90:95, 40:45] = 220
        tiles = []

        def infer(model, patches, *, batch_size):
            tiles.append(patches[0].copy())
            return [np.zeros((256, 256), np.float32)]

        with patch("src.PCE.inspection.predict_tiles", side_effect=infer):
            result = inspect_frame(frame, object(), 0.5, {})
        self.assertEqual(tiles[0][100, 100], 220)
        self.assertEqual(result.crops[0][100, 100, 0], 220)
        self.assertEqual(result.scores, (100.0, 100.0, 100.0))

    def test_black_crop_pixels_are_excluded_from_numerator_and_denominator(self):
        frame = sample_frame()
        frame[90:95, 40:45] = 0
        calls = 0

        def infer(model, patches, *, batch_size):
            nonlocal calls
            calls += 1
            probability = np.zeros((256, 256), np.float32)
            if calls == 1:
                probability[:128, :128] = 1
                probability[100, 100] = 1  # Inside the black part of P1.
            return [probability]

        with patch("src.PCE.inspection.predict_tiles", side_effect=infer):
            result = inspect_frame(frame, object(), 0.5, {})
        self.assertEqual(result.valid_pixels, (875, 900, 900))
        source_valid = np.any(frame[80:110, 30:60] != 0, axis=2)
        tile_exposure = np.zeros((256, 256), np.uint8)
        tile_exposure[:128, :128] = 1
        source_exposure = cv2.resize(tile_exposure, (30, 30), interpolation=cv2.INTER_NEAREST)
        expected = 100 * (1 - np.count_nonzero(source_exposure[source_valid]) / 875)
        self.assertAlmostEqual(result.scores[0], expected)
        np.testing.assert_array_equal(result.crops[0][100, 100], [0, 0, 0])
        np.testing.assert_array_equal(result.annotated_frame[92, 42], [0, 0, 0])

    def test_requires_exactly_three_regions_before_inference(self):
        with patch("src.PCE.inspection.predict_tiles") as predict:
            with self.assertRaisesRegex(ValueError, "정확히 3개.*2개"):
                inspect_frame(sample_frame(2), object(), 0.5, {})
        predict.assert_not_called()

    def test_history_reopens_full_crop_pce_results(self):
        frame = sample_frame()
        with patch("src.PCE.inspection.predict_tiles",
                   return_value=[np.zeros((256, 256), np.float32)]):
            result = inspect_frame(frame, object(), 0.5, {})
        with tempfile.TemporaryDirectory() as folder:
            store = InspectionHistoryStore(folder)
            record = InspectionRecord(*result.scores, measured_at="2026-10-26T12:00:00+09:00",
                                      source="file", metadata={"regions": result.regions,
                                      "conditions": {"inspection_mode": "certification"}})
            saved = store.save(record, frame, result.annotated_frame)
            event = gr.SelectData(None, {"index": [0, 0], "value": None, "selected": True})
            detail = StatisticsController(lambda: store).detail([saved.record_id], event)
            for reopened, displayed in zip(detail[3:6], result.crops, strict=True):
                np.testing.assert_array_equal(reopened, displayed)


if __name__ == "__main__":
    unittest.main()
