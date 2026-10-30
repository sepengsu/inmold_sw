"""Confidence is model certainty, with the exact scoring threshold and mask."""

import unittest

import numpy as np

from src.postprocess.confidence import prediction_confidence


class PredictionConfidenceTests(unittest.TestCase):
    def test_certain_and_ambiguous_predictions(self):
        self.assertEqual(prediction_confidence(np.array([[0., 1.]]), 0.5), 100.)
        self.assertEqual(prediction_confidence(np.array([[0.5, 0.5]]), 0.5), 50.)

    def test_respects_selected_label_at_nonstandard_threshold(self):
        self.assertAlmostEqual(prediction_confidence(np.array([[0.6, 0.8]]), 0.75), 60.)

    def test_nonblack_mask_excludes_unmeasured_pixels(self):
        p = np.array([[0.5, 0.9], [0.5, 0.1]])
        valid = np.array([[False, True], [False, True]])
        self.assertAlmostEqual(prediction_confidence(p, 0.5, valid), 90.)
        for invalid in (np.zeros((2, 2), bool), np.ones((1, 2), bool), np.ones((2, 2), np.uint8)):
            with self.assertRaises(ValueError):
                prediction_confidence(p, 0.5, invalid)

    def test_invalid_probability_cannot_become_a_confidence(self):
        for value in (float('nan'), -0.1, 1.1):
            with self.assertRaises(ValueError):
                prediction_confidence(np.array([[value]]), 0.5)
