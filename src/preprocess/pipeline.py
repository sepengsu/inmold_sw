"""Load -> grayscale -> native ROI -> CLAHE -> sliding tiles, without resizing."""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np

from ._validation import GrayImage, grayscale
from .clahe import apply_clahe
from .config import Config, validate_config
from .roi import crop_roi, valid_roi_mask
from .tiling import plan_tiling, tile_image

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


class RoiPreprocessResult(TypedDict):
    """Native ROI and enhancement, without allocating inference tiles."""

    original_shape: tuple[int, int]
    roi_shape: tuple[int, int]
    original: GrayImage
    roi: GrayImage
    valid_mask: np.ndarray
    clahe: GrayImage
    metadata: dict[str, Any]


class PreprocessResult(RoiPreprocessResult):
    """In-memory images, independent patches, and serializable merge metadata."""

    image_id: str
    patches: list[GrayImage]
    positions: list[tuple[int, int]]


def to_grayscale(image: np.ndarray) -> GrayImage:
    """Keep uint8 mono values unchanged, or convert OpenCV BGR/BGRA input."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.uint8:
        raise TypeError(f"image must have dtype uint8, got {image.dtype}")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.ndim == 3 and image.size and image.shape[2] in (3, 4):
        mode = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        image = cv2.cvtColor(image, mode)
    grayscale(image)
    return image


def load_image(path: str | Path) -> GrayImage:
    """Read uint8 grayscale with an imdecode fallback for Windows Unicode paths."""
    path = Path(path)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        encoded = np.fromfile(path, dtype=np.uint8)
        if encoded.size:
            image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return to_grayscale(image)


def _validate_image_id(image_id: str) -> None:
    """Require a portable filename component for all output names."""
    if (not isinstance(image_id, str) or not image_id or image_id in (".", "..")
            or any(c in '<>:"/\\|?*' or ord(c) < 32 for c in image_id)
            or image_id.endswith((".", " "))):
        raise ValueError(f"image_id must be a safe filename component, got {image_id!r}")
    if image_id.split(".")[0].upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise ValueError(f"image_id uses a reserved Windows filename: {image_id!r}")


def _metadata(
    path: Path, image_id: str, config: Config, original: GrayImage, roi: GrayImage,
    positions: list[tuple[int, int]], roi_detection: dict | None = None,
) -> dict[str, Any]:
    """Describe real ROI extents, padding and tile origins for a future merge."""
    size = config["tiling"]["patch_size"]
    layout = plan_tiling(roi.shape, **config["tiling"])
    height, width = roi.shape
    tiles = []
    for tile_id, (x, y) in enumerate(positions):
        valid_width, valid_height = min(size, width - x), min(size, height - y)
        filename = f"tile_{tile_id:04d}_x{x:04d}_y{y:04d}.png"
        tiles.append({
            "image_id": image_id, "tile_id": tile_id, "x": x, "y": y,
            "width": size, "height": size,
            "valid_width": valid_width, "valid_height": valid_height,
            "padding": {"top": 0, "bottom": size - valid_height,
                        "left": 0, "right": size - valid_width},
            "file": f"{image_id}/{filename}" if config["output"]["save_tiles"] else None,
        })
    metadata = {
        "schema_version": 1, "image_id": image_id, "source_path": str(path),
        "original_shape": list(original.shape), "roi_shape": list(roi.shape),
        "roi": dict(config["roi"]), "clahe": dict(config["clahe"]),
        "tiling": dict(config["tiling"]), "coordinate_order": "xy",
        "coordinate_space": "roi", "shape_order": "height_width",
        "padded_shape": list(layout.padded_shape),
        "padding": {**layout.padding, "mode": "reflect_101"}, "tiles": tiles,
    }
    if roi_detection is not None:
        metadata["roi_detection"] = roi_detection
    return metadata


def _roi_stage(original: GrayImage, config: Config) -> tuple:
    effective = dict(config)
    detection = None
    if config["auto_roi"]["enabled"]:
        from .auto_roi import detect_auto_roi

        result = detect_auto_roi(original, config["auto_roi"]["boundary_config"])
        effective["roi"] = result.bounds
        detection = result.metadata
    roi = crop_roi(original, **effective["roi"])
    metadata = {"original_shape": list(original.shape), "roi_shape": list(roi.shape),
                "roi": effective["roi"], "roi_detection": detection}
    valid_mask = valid_roi_mask(metadata)
    roi[~valid_mask] = 0
    options = config["clahe"]
    enhanced = (apply_clahe(roi, options["clip_limit"], options["tile_grid_size"])
                if options["enabled"] else roi.copy())
    enhanced[~valid_mask] = 0
    return effective, roi, enhanced, valid_mask, detection


def _save_outputs(result: PreprocessResult, config: Config) -> None:
    """Write exclusively; refuse existing files and roll back failed new writes."""
    output = config["output"]
    root = output["root_dir"]
    image_id = result["image_id"]
    images: list[tuple[Path, GrayImage]] = []
    if output["save_roi"]:
        images.append((root / "roi" / f"{image_id}_roi.png", result["roi"]))
    if output["save_clahe"]:
        images.append((root / "clahe" / f"{image_id}_clahe.png", result["clahe"]))
    if output["save_tiles"]:
        for patch, tile in zip(result["patches"], result["metadata"]["tiles"], strict=True):
            images.append((root / "tiles" / tile["file"], patch))
    metadata_path = root / "tiles" / f"{image_id}_tiles.json"
    targets = [path for path, _ in images]
    if output["save_metadata"]:
        targets.append(metadata_path)
    for path in targets:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}; use a new output.root_dir")
    created: list[Path] = []
    try:
        for path, pixels in images:
            ok, encoded = cv2.imencode(".png", pixels)
            if not ok:
                raise OSError(f"Cannot encode PNG: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                created.append(path)
                stream.write(encoded.tobytes())
        if output["save_metadata"]:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("x", encoding="utf-8") as stream:
                created.append(metadata_path)
                json.dump(result["metadata"], stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
    except (OSError, ValueError, TypeError, cv2.error):
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def preprocess_image(
    image_path: str | Path, config: Mapping[str, Any], *, image_id: str | None = None,
) -> PreprocessResult:
    """Preserve raw ROI, enhance it, tile at native resolution, and optionally save."""
    config = validate_config(config)
    path = Path(image_path).expanduser().resolve()
    image_id = path.stem if image_id is None else image_id
    _validate_image_id(image_id)
    original = load_image(path)
    config, roi, enhanced, valid_mask, detection = _roi_stage(original, config)
    patches, positions = tile_image(enhanced, **config["tiling"])
    result: PreprocessResult = {
        "image_id": image_id, "original_shape": original.shape, "roi_shape": roi.shape,
        "original": original, "roi": roi, "valid_mask": valid_mask, "clahe": enhanced,
        "patches": patches, "positions": positions,
        "metadata": _metadata(path, image_id, config, original, roi, positions, detection),
    }
    _save_outputs(result, config)
    return result


def preprocess_roi_array(image: np.ndarray, config: Mapping[str, Any]) -> RoiPreprocessResult:
    """Run the existing ROI/CLAHE stage without tiling the whole ROI or writing files."""
    config = validate_config(config)
    original = to_grayscale(image)
    config, roi, enhanced, valid_mask, detection = _roi_stage(original, config)
    metadata = {"original_shape": list(original.shape), "roi_shape": list(roi.shape),
                "roi": dict(config["roi"])}
    if detection is not None:
        metadata["roi_detection"] = detection
    return {"original_shape": original.shape, "roi_shape": roi.shape,
            "original": original, "roi": roi, "valid_mask": valid_mask,
            "clahe": enhanced, "metadata": metadata}


def preprocess_array(
    image: np.ndarray, config: Mapping[str, Any], *, image_id: str = "camera",
) -> PreprocessResult:
    """Preprocess an in-memory uint8 grayscale frame without writing files."""
    config = validate_config(config)
    _validate_image_id(image_id)
    original = to_grayscale(image)
    config, roi, enhanced, valid_mask, detection = _roi_stage(original, config)
    patches, positions = tile_image(enhanced, **config["tiling"])
    return {
        "image_id": image_id, "original_shape": original.shape, "roi_shape": roi.shape,
        "original": original, "roi": roi, "valid_mask": valid_mask, "clahe": enhanced,
        "patches": patches, "positions": positions,
        "metadata": _metadata(Path("<camera>"), image_id, config, original, roi, positions, detection),
    }
