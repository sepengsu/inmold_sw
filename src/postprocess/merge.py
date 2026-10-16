"""Merge overlapping tile probabilities using preprocessing metadata."""

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

from src.preprocess.roi import valid_roi_mask


def _shape(value: object, name: str) -> tuple[int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in value)):
        raise ValueError(f"{name} must be [height, width] with positive integers")
    return int(value[0]), int(value[1])


def merge_probabilities(
    probabilities: Sequence[np.ndarray], metadata: Mapping,
) -> np.ndarray:
    """Average overlap in ROI coordinates and discard reflected tile padding."""
    if metadata.get("coordinate_order") != "xy" or metadata.get("coordinate_space") != "roi":
        raise ValueError("Expected xy tile coordinates relative to the ROI")
    height, width = _shape(metadata.get("roi_shape"), "roi_shape")
    tiles = metadata.get("tiles")
    if not isinstance(tiles, list) or len(tiles) != len(probabilities) or not tiles:
        raise ValueError("Probability count must match non-empty metadata tiles")

    total = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.uint32)
    for probability, tile in zip(probabilities, tiles, strict=True):
        if not isinstance(tile, Mapping):
            raise ValueError("Each tile must be a mapping")
        try:
            x, y = tile["x"], tile["y"]
            valid_width, valid_height = tile["valid_width"], tile["valid_height"]
            tile_width, tile_height = tile["width"], tile["height"]
        except KeyError as exc:
            raise ValueError(f"Missing tile field: {exc.args[0]}") from exc
        fields = (x, y, valid_width, valid_height, tile_width, tile_height)
        if any(isinstance(n, bool) or not isinstance(n, Integral) for n in fields):
            raise ValueError("Tile coordinates and sizes must be integers")
        if (x < 0 or y < 0 or valid_width < 1 or valid_height < 1
                or valid_width > tile_width or valid_height > tile_height
                or x + valid_width > width or y + valid_height > height):
            raise ValueError("Tile extent is outside the ROI")
        if not isinstance(probability, np.ndarray) or probability.shape != (tile_height, tile_width):
            raise ValueError("Probability shape must match metadata tile size")
        if not np.issubdtype(probability.dtype, np.number):
            raise ValueError("Probabilities must be numeric")
        valid = probability[:valid_height, :valid_width]
        if not np.isfinite(valid).all() or np.any((valid < 0) | (valid > 1)):
            raise ValueError("Probabilities must be finite values in [0, 1]")
        total[y:y + valid_height, x:x + valid_width] += valid
        counts[y:y + valid_height, x:x + valid_width] += 1
    if np.any(counts == 0):
        raise ValueError("Tile predictions do not cover the full ROI")
    return (total / counts).astype(np.float32)


def threshold_mask(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return a boolean Exposure mask; threshold after overlap averaging."""
    if (not isinstance(probability, np.ndarray) or probability.ndim != 2
            or not probability.size or not np.issubdtype(probability.dtype, np.number)
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError("probability must be a finite 2D array in [0, 1]")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    return probability >= threshold


def restore_original_mask(roi_mask: np.ndarray, metadata: Mapping) -> np.ndarray:
    """Place a ROI mask in original image coordinates; outside ROI is False."""
    original_height, original_width = _shape(metadata.get("original_shape"), "original_shape")
    roi_height, roi_width = _shape(metadata.get("roi_shape"), "roi_shape")
    if not isinstance(roi_mask, np.ndarray) or roi_mask.shape != (roi_height, roi_width):
        raise ValueError("roi_mask shape does not match metadata ROI")
    roi = metadata.get("roi")
    if not isinstance(roi, Mapping):
        raise ValueError("metadata.roi must be a mapping")
    try:
        x, y, width, height = (roi[key] for key in ("x", "y", "width", "height"))
    except KeyError as exc:
        raise ValueError(f"Missing ROI field: {exc.args[0]}") from exc
    if (any(isinstance(n, bool) or not isinstance(n, Integral) for n in (x, y, width, height))
            or x < 0 or y < 0 or width != roi_width or height != roi_height
            or x + width > original_width or y + height > original_height):
        raise ValueError("ROI metadata is inconsistent with image shapes")
    mask = np.zeros((original_height, original_width), dtype=bool)
    mask[y:y + height, x:x + width] = roi_mask.astype(bool) & valid_roi_mask(metadata)
    return mask


def impregnation_pct(exposure_mask: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    """Compute 100 × (1 − Exposure pixel fraction) over the supplied mask."""
    if not isinstance(exposure_mask, np.ndarray) or exposure_mask.ndim != 2 or not exposure_mask.size:
        raise ValueError("exposure_mask must be a non-empty 2D array")
    if valid_mask is None:
        return 100.0 * (1.0 - np.count_nonzero(exposure_mask) / exposure_mask.size)
    if (not isinstance(valid_mask, np.ndarray) or valid_mask.dtype != np.bool_
            or valid_mask.shape != exposure_mask.shape or not valid_mask.any()):
        raise ValueError("valid_mask must be a nonempty boolean mask matching exposure_mask")
    return 100.0 * (1.0 - np.count_nonzero(exposure_mask[valid_mask]) / np.count_nonzero(valid_mask))
