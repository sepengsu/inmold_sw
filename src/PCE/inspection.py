"""Inspect three specimen windows contained by the white paper mask."""

from dataclasses import dataclass

import cv2
import numpy as np

from src.model import predict_tiles
from src.postprocess import impregnation_pct, threshold_mask
from src.postprocess.confidence import prediction_confidence
from src.preprocess import apply_clahe
from .detection import detect_regions, detect_windows, inset_region, oriented_region


TILE_SIZE = 256
# Native pixels per edge, applied before resampling to a 256x256 model tile.
ROI_INSET_PX = 3


@dataclass(frozen=True, slots=True)
class PceResult:
    scores: tuple[float, float, float]
    average: float
    annotated_frame: np.ndarray
    crops: tuple[np.ndarray, np.ndarray, np.ndarray]
    regions: tuple[dict, dict, dict]
    valid_pixels: tuple[int, int, int]
    defect_pixels: tuple[int, int, int]
    roi_detection: dict | None = None
    roi: None = None
    pce_detection: dict | None = None
    confidences: tuple[float, float, float] | None = None


def _rectification_matrix(region: dict) -> np.ndarray:
    width, height = region["rectified_width"], region["rectified_height"]
    destination = np.array(((0, 0), (width - 1, 0), (width - 1, height - 1),
                            (0, height - 1)), dtype=np.float32)
    return cv2.getPerspectiveTransform(np.asarray(region["quad"], dtype=np.float32), destination)


def rectify_region(image: np.ndarray, region: dict, *, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Sample one oriented rectangle in its native pixel grid."""
    return cv2.warpPerspective(image, _rectification_matrix(region),
                               (region["rectified_width"], region["rectified_height"]),
                               flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _project_to_bbox(image: np.ndarray, region: dict) -> np.ndarray:
    """Return an aligned mask in the original axis-aligned component bounds."""
    offset = np.array(((1, 0, -region["x"]), (0, 1, -region["y"]), (0, 0, 1)), dtype=np.float64)
    inverse = offset @ np.linalg.inv(_rectification_matrix(region))
    return cv2.warpPerspective(image, inverse, (region["width"], region["height"]),
                               flags=cv2.INTER_NEAREST)


def inspect_frame(frame_rgb: np.ndarray, model: object, threshold: float, config: dict) -> PceResult:
    """Use detection only for geometry; measure the complete original RGB rectangle."""
    if (not isinstance(frame_rgb, np.ndarray) or frame_rgb.dtype != np.uint8
            or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3 or not frame_rgb.size):
        raise ValueError("공인인증 입력은 uint8 RGB 영상이어야 합니다.")
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    detected = detect_windows(gray)
    annotated = frame_rgb.copy()
    scores, crops, regions, valid_pixels, defect_pixels = [], [], [], [], []
    confidences = []
    clahe = config.get("clahe", {})
    frame_valid = np.any(frame_rgb != 0, axis=2).astype(np.uint8)
    for point, (detected_region, _) in enumerate(detected.regions, start=1):
        region = inset_region(detected_region, ROI_INSET_PX)
        x, y, w, h = (region[key] for key in ("x", "y", "width", "height"))
        source_rgb = rectify_region(frame_rgb, region)
        # Threshold/component masks locate the rectangle only. Bright specimen
        # pixels on its boundary must remain in both the crop and its denominator.
        source_valid = rectify_region(frame_valid, region,
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
        if not source_valid.any():
            raise ValueError(f"P{point} 크롭에 검정색을 제외한 유효 픽셀이 없습니다.")
        source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
        if clahe.get("enabled", False):
            source_gray = apply_clahe(source_gray, clahe["clip_limit"], clahe["tile_grid_size"])
        tile = cv2.resize(source_gray, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
        tile_valid = cv2.resize(source_valid.astype(np.uint8), (TILE_SIZE, TILE_SIZE),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        tile[~tile_valid] = 0
        # Keep each region as an independent model invocation.
        probabilities = predict_tiles(model, [tile], batch_size=1)
        if len(probabilities) != 1 or probabilities[0].shape != tile.shape:
            raise ValueError(f"P{point} 추론 결과는 256×256 한 장이어야 합니다.")
        model_exposure = threshold_mask(probabilities[0], threshold)
        native_size = (region["rectified_width"], region["rectified_height"])
        source_exposure = cv2.resize(model_exposure.astype(np.uint8), native_size,
                                     interpolation=cv2.INTER_NEAREST).astype(bool) & source_valid
        # Nearest-neighbour matches the native label mask used by the score.
        source_probability = cv2.resize(probabilities[0], native_size, interpolation=cv2.INTER_NEAREST)
        confidences.append(prediction_confidence(source_probability, threshold, source_valid))
        scores.append(impregnation_pct(source_exposure, source_valid))
        valid_pixels.append(int(np.count_nonzero(source_valid)))
        defect_pixels.append(int(np.count_nonzero(source_exposure)))

        red = np.array([235, 64, 64], dtype=np.uint16)
        # Display the same native-resolution predictions used in the score/history.
        source_preview = source_rgb.copy()
        source_preview[source_exposure] = (
            (source_preview[source_exposure].astype(np.uint16) + red) // 2
        ).astype(np.uint8)
        preview = cv2.resize(source_preview, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
        crops.append(preview)

        destination = annotated[y:y + h, x:x + w]
        coverage = _project_to_bbox(np.ones(source_valid.shape, dtype=np.uint8), region).astype(bool)
        projected_exposure = (_project_to_bbox(source_exposure.astype(np.uint8), region).astype(bool)
                              & coverage & frame_valid[y:y + h, x:x + w].astype(bool))
        destination[projected_exposure] = (
            (destination[projected_exposure].astype(np.uint16) + red) // 2
        ).astype(np.uint8)
        regions.append(region)

    # The real paper contour is distinct from the individual specimen windows.
    outline = np.zeros(gray.shape, np.uint8)
    cv2.polylines(outline, [np.asarray(detected.paper.metadata["contour"], np.int32)], True, 1, 2)
    outline[:2] = outline[-2:] = 0
    outline[:, :2] = outline[:, -2:] = 0
    annotated[outline.astype(bool)] = (40, 220, 100)
    colors = ((255, 220, 0), (32, 215, 232), (234, 85, 223))
    for point, (region, color) in enumerate(zip(regions, colors, strict=True), start=1):
        x, y, w, h = (region[key] for key in ("x", "y", "width", "height"))
        # Offset the rotated outline outside its crop so history detail images stay clean.
        quad = np.asarray(region["quad"], np.float32)
        horizontal = (quad[1] - quad[0]) / np.linalg.norm(quad[1] - quad[0])
        vertical = (quad[3] - quad[0]) / np.linalg.norm(quad[3] - quad[0])
        offsets = np.array((-horizontal - vertical, horizontal - vertical,
                            horizontal + vertical, -horizontal + vertical)) * 3
        display = np.rint(quad + offsets).astype(np.int32)
        cv2.polylines(annotated, [display], True, color, 2)
        label_y = y - 8 if y >= 28 else min(frame_rgb.shape[0] - 4, y + h + 18)
        cv2.putText(annotated, f"P{point}", (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return PceResult(
        scores=tuple(scores), average=float(np.mean(scores)), annotated_frame=annotated,
        crops=tuple(crops), regions=tuple(regions), valid_pixels=tuple(valid_pixels),
        defect_pixels=tuple(defect_pixels),
        confidences=tuple(confidences),
        pce_detection={**detected.metadata, "measurement_geometry": "original_rectangle",
                       "valid_pixel_rule": "original_rgb_nonblack", "roi_inset_px": ROI_INSET_PX},
    )
