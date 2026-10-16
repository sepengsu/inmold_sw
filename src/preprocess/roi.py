"""Fixed-coordinate cropping at the original pixel resolution."""

from ._validation import GrayImage, grayscale, integer

import numpy as np


def crop_roi(image: GrayImage, x: int, y: int, width: int, height: int) -> GrayImage:
    """Copy a valid (x, y, width, height) ROI without changing its pixels."""
    grayscale(image)
    x, y = integer(x, "roi.x", 0), integer(y, "roi.y", 0)
    width = integer(width, "roi.width")
    height = integer(height, "roi.height")
    image_height, image_width = image.shape
    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI (x={x}, y={y}, width={width}, height={height}) exceeds "
            f"image bounds (width={image_width}, height={image_height})"
        )
    return image[y:y + height, x:x + width].copy()


def valid_roi_mask(metadata: dict) -> np.ndarray:
    """Reconstruct the exact fabric mask inside a native-coordinate ROI box."""
    height, width = metadata["roi_shape"]
    geometry = metadata.get("roi_detection")
    if geometry is None:
        return np.ones((height, width), dtype=bool)
    original_height, original_width = metadata["original_shape"]
    if geometry.get("source_shape") != [original_height, original_width]:
        raise ValueError("ROI boundary source shape does not match the image")
    top = np.asarray(geometry.get("top_y"))
    bottom = np.asarray(geometry.get("bottom_y_exclusive"))
    if (top.shape != (original_width,) or bottom.shape != top.shape
            or not np.issubdtype(top.dtype, np.integer)
            or not np.issubdtype(bottom.dtype, np.integer)
            or np.any(top < 0) or np.any(bottom > original_height)
            or np.any(top >= bottom)):
        raise ValueError("Invalid native-pixel ROI boundaries")
    roi = metadata["roi"]
    x, y = roi["x"], roi["y"]
    if (roi["width"] != width or roi["height"] != height or x < 0 or y < 0
            or x + width > original_width or y + height > original_height):
        raise ValueError("ROI box is inconsistent with the image")
    top, bottom = top[x:x + width], bottom[x:x + width]
    if np.any(top < y) or np.any(bottom > y + height):
        raise ValueError("ROI box clips the detected fabric boundaries")
    rows = np.arange(y, y + height)[:, None]
    return (rows >= top) & (rows < bottom)
