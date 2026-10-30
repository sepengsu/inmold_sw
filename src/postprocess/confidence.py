"""Mean probability assigned to the threshold-selected binary pixel label."""

import numpy as np

from .merge import threshold_mask


CONFIDENCE_METHOD = "mean_threshold_selected_binary_label_probability_percent"


def prediction_confidence(probability: np.ndarray, threshold: float,
                          valid_mask: np.ndarray | None = None) -> float:
    """Return 0–100 model certainty, not calibrated accuracy or ROI quality.

    Exposure/Weldline is the positive binary class used by the score. Each
    pixel contributes p when classified positive and 1-p otherwise. Certification
    supplies its native nonblack mask so confidence and score use the same pixels.
    """
    positive = threshold_mask(probability, threshold)
    certainty = np.where(positive, probability, 1.0 - probability)
    if valid_mask is not None:
        if (not isinstance(valid_mask, np.ndarray) or valid_mask.dtype != np.bool_
                or valid_mask.shape != probability.shape or not valid_mask.any()):
            raise ValueError("confidence valid_mask must contain valid boolean pixels")
        certainty = certainty[valid_mask]
    return float(np.mean(certainty, dtype=np.float64) * 100.0)
