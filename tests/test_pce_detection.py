"""Paper containment, window layout, and masked scoring regressions."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.PCE.detection import detect_paper_mask, detect_windows, inset_region, oriented_region, rectangle_inside_mask
from src.PCE.inspection import inspect_frame, rectify_region


def paper_frame():
    gray = np.full((300, 650), 40, np.uint8)
    gray[50:250, 30:520] = 240
    for x in (80, 230, 380):
        gray[130:170, x:x + 40] = 100
    return gray


@patch("src.PCE.inspection.ROI_INSET_PX", 0)
class PaperWindowTests(unittest.TestCase):
    """Check untrimmed geometry/pixel preservation independently of the default inset."""

    def test_paper_envelope_fills_windows_and_excludes_background(self):
        gray = paper_frame()
        paper = detect_paper_mask(gray)
        self.assertTrue(paper.interior[150, 100])
        self.assertFalse(paper.mask[150, 600])
        self.assertFalse(paper.interior[55, 100])
        self.assertTrue(paper.interior[70, 100])

    def test_external_square_noise_cannot_become_a_window(self):
        gray = paper_frame()
        gray[75:240, 550:640] = 200
        gray[100:130, 570:600] = 100
        gray[190:220, 580:610] = 100
        result = detect_windows(gray)
        self.assertEqual([r["x"] for r, _ in result.regions], [80, 230, 380])
        self.assertEqual(result.metadata["candidate_count"], 3)
        for region, _ in result.regions:
            self.assertTrue(rectangle_inside_mask(region, result.paper.interior))

    def test_paper_clipped_at_image_edge_is_supported(self):
        gray = paper_frame()
        gray[50:250, :30] = 240
        result = detect_windows(gray)
        self.assertEqual(result.paper.metadata["bounds"]["x"], 0)
        self.assertEqual([r["x"] for r, _ in result.regions], [80, 230, 380])

    def test_extra_internal_candidate_is_resolved_by_three_point_layout(self):
        gray = paper_frame()
        gray[70:100, 460:490] = 100
        result = detect_windows(gray)
        self.assertEqual(result.metadata["candidate_count"], 4)
        self.assertEqual([r["x"] for r, _ in result.regions], [80, 230, 380])

    def test_equally_plausible_triplets_are_rejected_before_inference(self):
        frame = np.full((300, 700, 3), 240, np.uint8)
        for x in (80, 230, 380, 530):
            frame[130:170, x:x + 40] = 100
        with patch("src.PCE.inspection.predict_tiles") as predict:
            with self.assertRaisesRegex(ValueError, "선택이 모호"):
                inspect_frame(frame, object(), 0.5, {})
        predict.assert_not_called()

    def test_missing_window_cannot_be_replaced_by_external_noise(self):
        gray = paper_frame()
        gray[130:170, 380:420] = 240
        gray[75:240, 550:640] = 200
        gray[140:180, 570:610] = 100
        with self.assertRaisesRegex(ValueError, "종이 내부.*2개"):
            detect_windows(gray)

    def test_small_internal_noise_cannot_replace_a_missing_window(self):
        gray = paper_frame()
        gray[130:170, 380:420] = 240
        gray[130:155, 380:405] = 100
        with self.assertRaisesRegex(ValueError, "크기·정렬·간격"):
            detect_windows(gray)

    def test_window_connected_to_outside_of_paper_is_rejected(self):
        gray = paper_frame()
        gray[50:140, 395:415] = 40
        with self.assertRaisesRegex(ValueError, "종이 내부.*2개"):
            detect_windows(gray)

    def test_entire_rotated_rectangle_must_fit_not_only_its_corners(self):
        mask = np.ones((100, 100), bool)
        mask[:55, 45:55] = False
        region = {"quad": [[20, 20], [80, 20], [80, 80], [20, 80]]}
        self.assertTrue(all(mask[y, x] for x, y in region["quad"]))
        self.assertFalse(rectangle_inside_mask(region, mask))

    def test_tilted_three_point_row_is_supported(self):
        gray = np.full((300, 650), 240, np.uint8)
        for center in ((100, 90), (300, 140), (500, 190)):
            quad = cv2.boxPoints((center, (40, 40), 20))
            cv2.fillConvexPoly(gray, np.rint(quad).astype(np.int32), 100)
        result = detect_windows(gray)
        self.assertEqual(len(result.regions), 3)
        self.assertLess(result.metadata["layout_score"], 0.02)

    def test_oriented_rectangle_keeps_original_image_coordinates_and_pixel_extents(self):
        region = {"x": 123, "y": 456, "width": 41, "height": 31, "area": 1271}
        result = oriented_region(region, np.ones((31, 41), bool))
        np.testing.assert_allclose(result["quad"], [[123, 456], [163, 456], [163, 486], [123, 486]])
        self.assertEqual((result["rectified_width"], result["rectified_height"]), (41, 31))
        self.assertEqual(result["rotation_deg"], 0)

    def test_positive_and_negative_tilts_restore_upright_original_rgb_rois(self):
        frame = np.full((500, 900, 3), 240, np.uint8)
        colors = ((100, 30, 30), (30, 100, 30), (30, 30, 100), (110, 90, 30))
        for center_x in (200, 450, 700):
            x, y = center_x - 38, 212
            frame[y:y + 77, x:x + 77] = 100
            for (dx, dy), color in zip(((8, 8), (57, 8), (8, 57), (57, 57)), colors, strict=True):
                frame[y + dy:y + dy + 12, x + dx:x + dx + 12] = color
            # Bright pixels open to the upper boundary must survive extraction.
            frame[y:y + 8, x + 29:x + 47] = 220
        for angle in (-30, -12, 0, 12, 30):
            with self.subTest(angle=angle):
                matrix = cv2.getRotationMatrix2D((450, 250), -angle, 1)
                tilted = cv2.warpAffine(frame, matrix, (900, 500), borderValue=(240, 240, 240))
                original = tilted.copy()
                with patch("src.PCE.inspection.predict_tiles",
                           return_value=[np.zeros((256, 256), np.float32)]):
                    result = inspect_frame(tilted, object(), 0.5, {"clahe": {"enabled": False}})
                self.assertEqual(len(result.regions), 3)
                for region, preview in zip(result.regions, result.crops, strict=True):
                    self.assertAlmostEqual(region["rotation_deg"], angle, delta=1)
                    self.assertAlmostEqual(region["rectified_width"], 77, delta=2)
                    self.assertAlmostEqual(region["rectified_height"], 77, delta=2)
                    for (px, py), color in zip(((48, 48), (208, 48), (48, 208), (208, 208)),
                                                colors, strict=True):
                        np.testing.assert_allclose(preview[py, px], color, atol=2)
                    self.assertGreater(int(preview[8, 128, 0]), 180)
                    expected = cv2.resize(rectify_region(original, region), (256, 256),
                                          interpolation=cv2.INTER_LINEAR)
                    np.testing.assert_array_equal(preview, expected)
                    self.assertFalse(np.any(np.all(preview == 0, axis=2)))
                np.testing.assert_array_equal(tilted, original)

    def test_no_white_paper_reports_detection_failure(self):
        for gray in (np.zeros((300, 650), np.uint8), np.full((300, 650), 100, np.uint8)):
            with self.subTest(value=int(gray[0, 0])):
                with self.assertRaisesRegex(ValueError, "흰 종이 마스크"):
                    detect_windows(gray)

    def test_original_rectangle_including_bright_boundary_pixels_is_scored(self):
        frame = np.full((180, 320, 3), 240, np.uint8)
        for x in (30, 130, 230):
            frame[80:110, x:x + 30] = 100
        frame[80:88, 30:38] = 240  # Bright pixels open to the component's outer boundary.
        frame[90:95, 40:45] = 220  # Bright specimen pixels inside that window.
        tiles = []

        def infer(model, patches, *, batch_size):
            tiles.append(patches[0].copy())
            return [np.ones((256, 256), np.float32)]

        with patch("src.PCE.inspection.predict_tiles", side_effect=infer):
            result = inspect_frame(frame, object(), 0.5, {})
        self.assertEqual(tiles[0][10, 10], 240)
        self.assertEqual(tiles[0][100, 100], 220)
        self.assertEqual(result.valid_pixels, (900, 900, 900))
        self.assertEqual(result.defect_pixels, result.valid_pixels)
        self.assertEqual(result.scores, (0.0, 0.0, 0.0))
        np.testing.assert_array_equal(result.crops[0][10, 10], [237, 152, 152])
        np.testing.assert_array_equal(result.annotated_frame[82, 32], [237, 152, 152])

    def test_detection_contour_does_not_erase_any_original_crop_pixels(self):
        frame = np.full((180, 320, 3), 240, np.uint8)
        for x in (30, 130, 230):
            frame[80:110, x:x + 30] = 100
            frame[80:86, x + 4:x + 16] = 240
            frame[97:104, x:x + 6] = 220
        original = frame.copy()
        tiles = []

        def infer(model, patches, *, batch_size):
            tiles.append(patches[0].copy())
            return [np.zeros((256, 256), np.float32)]

        with patch("src.PCE.inspection.predict_tiles", side_effect=infer):
            result = inspect_frame(frame, object(), 0.5, {"clahe": {"enabled": False}})
        for region, tile, preview, total in zip(result.regions, tiles, result.crops,
                                                result.valid_pixels, strict=True):
            crop = rectify_region(original, region)
            expected = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            np.testing.assert_array_equal(preview, expected)
            np.testing.assert_array_equal(tile, cv2.cvtColor(expected, cv2.COLOR_RGB2GRAY))
            self.assertFalse(np.any(np.all(preview == 0, axis=2)))
            self.assertEqual(total, region["rectified_width"] * region["rectified_height"])
        np.testing.assert_array_equal(frame, original)


class RoiInsetTests(unittest.TestCase):
    def test_default_inset_removes_border_from_model_preview_score_and_overlay(self):
        frame = np.full((180, 320, 3), 240, np.uint8)
        for x in (30, 130, 230):
            frame[80:110, x:x + 30] = 100
            frame[80:82, x + 5:x + 25] = 220
            frame[108:110, x + 5:x + 25] = 220
        original = frame.copy()
        tiles = []

        def infer(model, patches, *, batch_size):
            tiles.append(patches[0].copy())
            return [np.ones((256, 256), np.float32)]

        with patch("src.PCE.inspection.predict_tiles", side_effect=infer):
            result = inspect_frame(frame, object(), 0.5, {"clahe": {"enabled": False}})
        self.assertEqual(result.pce_detection["roi_inset_px"], 3)
        self.assertEqual(result.valid_pixels, (24 * 24,) * 3)
        self.assertEqual(result.defect_pixels, result.valid_pixels)
        for tile, preview, region in zip(tiles, result.crops, result.regions, strict=True):
            self.assertTrue(np.all(tile == 100))
            self.assertTrue(np.all(preview == (167, 82, 82)))
            self.assertEqual((region["rectified_width"], region["rectified_height"]), (24, 24))
            self.assertEqual((region["source_rectified_width"], region["source_rectified_height"]), (30, 30))
        # This source pixel lies between the measurement and its display outline.
        np.testing.assert_array_equal(result.annotated_frame[82, 45], original[82, 45])
        np.testing.assert_array_equal(result.annotated_frame[85, 45], [167, 82, 82])
        np.testing.assert_array_equal(frame, original)

    def test_rotated_inset_preserves_angle_center_and_original_rgb(self):
        gray = np.full((250, 550), 240, np.uint8)
        for x in (100, 275, 450):
            quad = cv2.boxPoints(((x, 125), (76, 76), 22))
            cv2.fillConvexPoly(gray, np.rint(quad).astype(np.int32), 100)
        detection = detect_windows(gray)
        frame = np.repeat(gray[:, :, None], 3, axis=2)
        with patch("src.PCE.inspection.predict_tiles", return_value=[np.zeros((256, 256), np.float32)]):
            result = inspect_frame(frame, object(), 0.5, {})
        for (source, _), region, preview in zip(detection.regions, result.regions, result.crops, strict=True):
            np.testing.assert_allclose(np.mean(region["quad"], axis=0),
                                       np.mean(source["quad"], axis=0), atol=1e-4)
            self.assertEqual(region["rotation_deg"], source["rotation_deg"])
            self.assertEqual(region["rectified_width"], source["rectified_width"] - 6)
            self.assertEqual(region["rectified_height"], source["rectified_height"] - 6)
            expected = cv2.resize(rectify_region(frame, region), (256, 256), interpolation=cv2.INTER_LINEAR)
            np.testing.assert_array_equal(preview, expected)
            self.assertTrue(np.all(preview == 100))

    def test_invalid_or_excessive_inset_is_rejected(self):
        region = oriented_region({"x": 20, "y": 20, "width": 10, "height": 10}, np.ones((10, 10), bool))
        for margin in (-1, 1.5, True, 5):
            with self.subTest(margin=margin):
                with self.assertRaises(ValueError):
                    inset_region(region, margin)


if __name__ == "__main__":
    unittest.main()
