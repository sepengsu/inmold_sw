"""Sliding crops with complete ROI coverage and minimal reflection padding."""

from dataclasses import dataclass

import cv2

from ._validation import GrayImage, grayscale, integer


@dataclass(frozen=True)
class TilingLayout:
    """ROI-relative (x, y) coordinates and padding on the bottom/right only."""

    positions: list[tuple[int, int]]
    padded_shape: tuple[int, int]
    padding: dict[str, int]


def validate_tiling(patch_size: int, stride: int) -> tuple[int, int]:
    """Reject invalid dimensions and strides that would leave coverage gaps."""
    patch_size = integer(patch_size, "tiling.patch_size")
    stride = integer(stride, "tiling.stride")
    if stride > patch_size:
        raise ValueError("tiling.stride must be <= patch_size to cover every ROI pixel")
    return patch_size, stride


def _axis_positions(length: int, patch_size: int, stride: int) -> list[int]:
    """Append the end-aligned crop only when regular strides miss it."""
    last = max(0, length - patch_size)
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def plan_tiling(
    image_shape: tuple[int, int], patch_size: int = 256, stride: int = 128,
) -> TilingLayout:
    """Compute row-major crop positions without discarding any border pixels."""
    patch_size, stride = validate_tiling(patch_size, stride)
    if len(image_shape) != 2:
        raise ValueError("image_shape must be (height, width)")
    height = integer(image_shape[0], "image height")
    width = integer(image_shape[1], "image width")
    bottom, right = max(0, patch_size - height), max(0, patch_size - width)
    positions = [(x, y) for y in _axis_positions(height, patch_size, stride)
                 for x in _axis_positions(width, patch_size, stride)]
    return TilingLayout(positions, (height + bottom, width + right),
                        {"top": 0, "bottom": bottom, "left": 0, "right": right})


def tile_image(
    image: GrayImage, patch_size: int = 256, stride: int = 128,
) -> tuple[list[GrayImage], list[tuple[int, int]]]:
    """Return independent patches and (x, y) origins relative to the raw ROI."""
    grayscale(image)
    layout = plan_tiling(image.shape, patch_size, stride)
    padding = layout.padding
    if padding["bottom"] or padding["right"]:
        # REFLECT_101 also handles one-pixel axes by repeating their only value.
        image = cv2.copyMakeBorder(image, 0, padding["bottom"], 0, padding["right"],
                                   borderType=cv2.BORDER_REFLECT_101)
    patches = [image[y:y + patch_size, x:x + patch_size].copy()
               for x, y in layout.positions]
    return patches, layout.positions
