"""CLAHE for an already cropped, uint8 grayscale ROI."""

import cv2

from ._validation import GrayImage, grayscale, integer, positive_float


def validate_clahe(
    clip_limit: float, tile_grid_size: tuple[int, int] | list[int],
) -> tuple[float, tuple[int, int]]:
    """Validate CLAHE parameters before passing them to OpenCV."""
    clip = positive_float(clip_limit, "clahe.clip_limit")
    if not isinstance(tile_grid_size, (tuple, list)) or len(tile_grid_size) != 2:
        raise ValueError("clahe.tile_grid_size must contain two positive integers")
    grid = (integer(tile_grid_size[0], "clahe.tile_grid_size[0]"),
            integer(tile_grid_size[1], "clahe.tile_grid_size[1]"))
    return clip, grid


def apply_clahe(
    image: GrayImage,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] | list[int] = (8, 8),
) -> GrayImage:
    """Return a new uint8 image with the same shape as the supplied ROI."""
    grayscale(image)
    clip, grid = validate_clahe(clip_limit, tile_grid_size)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(image)
