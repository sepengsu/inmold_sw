"""Shared validation for lossless uint8 preprocessing operations."""

from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]


def integer(value: object, name: str, minimum: int = 1) -> int:
    """Require an integer without silently accepting floats or booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def positive_float(value: object, name: str) -> float:
    """Require a finite positive real parameter."""
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return float(value)


def grayscale(image: GrayImage) -> None:
    """Reject empty, non-grayscale, or non-uint8 arrays without casting."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"image must be a non-empty 2D grayscale array, got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"image must have dtype uint8, got {image.dtype}")
