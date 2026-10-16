"""Checks for ROI/tile reconstruction and reference U-Net checkpoint loading."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.preprocess import preprocess_image
from src.postprocess import merge_probabilities, restore_original_mask, threshold_mask


class PipelineTests(unittest.TestCase):
    def test_preprocess_and_merge_preserve_roi_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            image = np.arange(9 * 11, dtype=np.uint8).reshape(9, 11)
            self.assertTrue(cv2.imwrite(str(path), image))
            config = {
                "roi": {"x": 2, "y": 1, "width": 7, "height": 5},
                "clahe": {"enabled": False},
                "tiling": {"patch_size": 4, "stride": 2},
                "output": {"root_dir": folder, "save_roi": False, "save_clahe": False,
                           "save_tiles": False, "save_metadata": False},
            }
            result = preprocess_image(path, config)
            truth = result["roi"].astype(np.float32) / 255
            tiles = []
            for tile in result["metadata"]["tiles"]:
                x, y = tile["x"], tile["y"]
                height, width = tile["valid_height"], tile["valid_width"]
                probability = np.full((4, 4), 0.99, dtype=np.float32)
                probability[:height, :width] = truth[y:y + height, x:x + width]
                tiles.append(probability)
            merged = merge_probabilities(tiles, result["metadata"])
            np.testing.assert_allclose(merged, truth, rtol=0, atol=1e-7)
            original = restore_original_mask(threshold_mask(merged, 0.2), result["metadata"])
            expected = np.zeros_like(image, dtype=bool)
            expected[1:6, 2:9] = truth >= 0.2
            np.testing.assert_array_equal(original, expected)

    def test_reflected_padding_is_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "small.png"
            self.assertTrue(cv2.imwrite(str(path), np.full((2, 3), 100, dtype=np.uint8)))
            config = {
                "roi": {"x": 0, "y": 0, "width": 3, "height": 2},
                "clahe": {"enabled": False},
                "tiling": {"patch_size": 4, "stride": 2},
                "output": {"root_dir": folder, "save_roi": False, "save_clahe": False,
                           "save_tiles": False, "save_metadata": False},
            }
            result = preprocess_image(path, config)
            probability = np.ones((4, 4), dtype=np.float32)
            probability[:2, :3] = 0.25
            np.testing.assert_array_equal(
                merge_probabilities([probability], result["metadata"]),
                np.full((2, 3), 0.25, dtype=np.float32),
            )


class ModelTests(unittest.TestCase):
    def test_checkpoint_and_prediction(self):
        import torch
        from src.model import UNet, load_checkpoint, predict_tiles

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best.pt"
            model = UNet(base_channels=8)
            torch.save({
                "architecture": "grayscale_unet", "base_channels": 8,
                "model_state_dict": model.state_dict(), "epoch": 1,
                "threshold": 0.5, "label": "Exposure", "mask_rule": "alpha > 0",
                "normalization": "uint8 / 255",
            }, path)
            loaded, info = load_checkpoint(path)
            self.assertEqual(info["threshold"], 0.5)
            outputs = predict_tiles(loaded, [np.zeros((32, 32), dtype=np.uint8)])
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0].shape, (32, 32))
            self.assertTrue(np.isfinite(outputs[0]).all())
            self.assertTrue(((0 <= outputs[0]) & (outputs[0] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
