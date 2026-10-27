"""Find three dark regions, straighten each one, and infer full 256px tiles."""

from dataclasses import dataclass

import cv2
import numpy as np

from src.model import predict_tiles
from src.postprocess import impregnation_pct, threshold_mask
from src.preprocess import apply_clahe


TILE_SIZE = 256
THRESHOLD = 150
MIN_AREA = 500
MAX_AREA = 20000
EDGE_MARGIN = 10


@dataclass(frozen=True, slots=True)
class PceResult:
    scores: tuple[float, float, float]
    average: float
    annotated_frame: np.ndarray
    crops: tuple[np.ndarray, np.ndarray, np.ndarray]
    regions: tuple[dict, dict, dict]
    valid_pixels: tuple[int, int, int]
    roi_detection: None = None
    roi: None = None


def detect_regions(gray: np.ndarray) -> tuple[tuple[dict, np.ndarray], ...]:
    """Apply the PCE threshold and connected-component filters in source pixels."""
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
        raise ValueError("공인인증 입력은 비어 있지 않은 uint8 흑백 영상이어야 합니다.")
    height, width = gray.shape
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    regions = []
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if not MIN_AREA <= area <= MAX_AREA or not 0.7 <= w / h <= 1.3:
            continue
        if x <= EDGE_MARGIN or y <= EDGE_MARGIN:
            continue
        if x + w >= width - EDGE_MARGIN or y + h >= height - EDGE_MARGIN:
            continue
        region = {"x": x, "y": y, "width": w, "height": h, "area": area}
        regions.append((region, labels[y:y + h, x:x + w] == index))
    regions.sort(key=lambda item: (item[0]["x"], item[0]["y"]))
    if len(regions) != 3:
        raise ValueError(f"공인인증 영역은 정확히 3개여야 합니다. 검출된 영역: {len(regions)}개")
    return tuple(regions)


def oriented_region(region: dict, component: np.ndarray) -> dict:
    """Fit the smallest rotated rectangle around a connected component."""
    y_pixels, x_pixels = np.nonzero(component)
    points = np.column_stack((x_pixels + region["x"], y_pixels + region["y"])).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(points))
    top, bottom = box[np.argsort(box[:, 1])[:2]], box[np.argsort(box[:, 1])[2:]]
    top_left, top_right = top[np.argsort(top[:, 0])]
    bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]
    quad = np.array((top_left, top_right, bottom_right, bottom_left), dtype=np.float32)
    rectified_width = max(2, round(float(np.linalg.norm(top_right - top_left))) + 1)
    rectified_height = max(2, round(float(np.linalg.norm(bottom_left - top_left))) + 1)
    angle = float(np.degrees(np.arctan2(top_right[1] - top_left[1],
                                        top_right[0] - top_left[0])))
    return {**region, "quad": quad.tolist(), "rectified_width": rectified_width,
            "rectified_height": rectified_height, "rotation_deg": angle}


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
    """Infer straightened crops and score their nonblack native pixels only."""
    if (not isinstance(frame_rgb, np.ndarray) or frame_rgb.dtype != np.uint8
            or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3 or not frame_rgb.size):
        raise ValueError("공인인증 입력은 uint8 RGB 영상이어야 합니다.")
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    detected = detect_regions(gray)
    annotated = np.zeros_like(frame_rgb)
    scores, crops, regions, valid_pixels = [], [], [], []
    clahe = config.get("clahe", {})
    frame_valid = np.any(frame_rgb != 0, axis=2).astype(np.uint8)
    for point, (bounds, component) in enumerate(detected, start=1):
        region = oriented_region(bounds, component)
        x, y, w, h = (region[key] for key in ("x", "y", "width", "height"))
        source_rgb = rectify_region(frame_rgb, region)
        source_valid = rectify_region(frame_valid, region, interpolation=cv2.INTER_NEAREST).astype(bool)
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
        exposure = model_exposure & tile_valid
        native_size = (region["rectified_width"], region["rectified_height"])
        source_exposure = cv2.resize(model_exposure.astype(np.uint8), native_size,
                                     interpolation=cv2.INTER_NEAREST).astype(bool) & source_valid
        scores.append(impregnation_pct(source_exposure, source_valid))
        valid_pixels.append(int(np.count_nonzero(source_valid)))

        preview = cv2.resize(source_rgb, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
        preview[~tile_valid] = 0
        red = np.array([235, 64, 64], dtype=np.uint16)
        preview[exposure] = (
            (preview[exposure].astype(np.uint16) + red) // 2
        ).astype(np.uint8)
        crops.append(preview)

        destination = annotated[y:y + h, x:x + w]
        coverage = _project_to_bbox(np.ones(source_valid.shape, dtype=np.uint8), region).astype(bool)
        destination[coverage] = frame_rgb[y:y + h, x:x + w][coverage]
        projected_exposure = (_project_to_bbox(source_exposure.astype(np.uint8), region).astype(bool)
                              & coverage & frame_valid[y:y + h, x:x + w].astype(bool))
        destination[projected_exposure] = (
            (destination[projected_exposure].astype(np.uint16) + red) // 2
        ).astype(np.uint8)
        regions.append(region)
    return PceResult(tuple(scores), float(np.mean(scores)), annotated,
                     tuple(crops), tuple(regions), tuple(valid_pixels))
