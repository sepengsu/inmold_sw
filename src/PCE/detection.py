"""Detect three specimen windows strictly inside a white paper mask."""

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np


THRESHOLD = 150
MIN_AREA = 500
MAX_AREA = 20000
EDGE_MARGIN = 10
MIN_RECTANGULARITY = 0.65
MAX_SIZE_RATIO = 1.8
MAX_ALIGNMENT_ERROR = 0.2  # Fraction of the typical window side length.
MAX_SPACING_RATIO = 1.35
AMBIGUITY_GAP = 0.1
MAX_CANDIDATES = 32


@dataclass(frozen=True, slots=True)
class PaperMask:
    mask: np.ndarray
    interior: np.ndarray
    metadata: dict


@dataclass(frozen=True, slots=True)
class WindowDetection:
    paper: PaperMask
    regions: tuple[tuple[dict, np.ndarray], ...]
    metadata: dict


def detect_paper_mask(gray: np.ndarray) -> PaperMask:
    """Fill the paper's actual outer contour, including all specimen windows."""
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
        raise ValueError("공인인증 입력은 비어 있지 않은 uint8 흑백 영상이어야 합니다.")
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold, bright = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if count <= 1:
        raise ValueError("공인인증 흰 종이 마스크를 검출하지 못했습니다.")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == index).astype(np.uint8)
    if np.median(gray[component.astype(bool)]) <= THRESHOLD:
        raise ValueError("공인인증 흰 종이 마스크의 밝기가 부족합니다. 조명과 노출을 확인하세요.")
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED)
    (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
    area = int(np.count_nonzero(mask))
    if (area < max(3 * MIN_AREA, gray.size * 0.1)
            or area / max(1, (rect_width + 1) * (rect_height + 1)) < 0.7):
        raise ValueError("공인인증 흰 종이 마스크의 면적 또는 외곽 형태가 올바르지 않습니다.")
    interior = cv2.erode(mask, np.ones((2 * EDGE_MARGIN + 1,) * 2, np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if not interior.any():
        raise ValueError("공인인증 흰 종이 마스크 내부의 검사 가능 영역이 부족합니다.")
    # A compact display contour; containment always uses the full-resolution mask.
    display_contour = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
    return PaperMask(mask.astype(bool), interior.astype(bool), {
        "method": "otsu_external_contour", "threshold": float(threshold),
        "source_shape": list(gray.shape), "contour": display_contour.tolist(),
        "bounds": dict(zip(("x", "y", "width", "height"), map(int, cv2.boundingRect(contour)))),
        "pixels": area, "edge_margin": EDGE_MARGIN,
    })


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


def inset_region(region: dict, margin: int) -> dict:
    """Move each edge inward along the rotated rectangle's own source-pixel axes."""
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("ROI 안쪽 여백은 0 이상의 정수 픽셀이어야 합니다.")
    width = region["rectified_width"] - 2 * margin
    height = region["rectified_height"] - 2 * margin
    if width < 2 or height < 2:
        raise ValueError(f"ROI를 각 변에서 {margin}px 줄이기에는 크기가 부족합니다.")
    quad = np.asarray(region["quad"], np.float32)
    horizontal = quad[1] - quad[0]
    vertical = quad[3] - quad[0]
    if np.linalg.norm(horizontal) <= 2 * margin or np.linalg.norm(vertical) <= 2 * margin:
        raise ValueError("ROI 안쪽 여백을 적용할 수 없는 사각형입니다.")
    horizontal = horizontal / np.linalg.norm(horizontal)
    vertical = vertical / np.linalg.norm(vertical)
    offsets = np.array((horizontal + vertical, -horizontal + vertical,
                        -horizontal - vertical, horizontal - vertical)) * margin
    return {**region, "source_quad": quad.tolist(),
            "source_rectified_width": region["rectified_width"],
            "source_rectified_height": region["rectified_height"],
            "quad": (quad + offsets).tolist(), "rectified_width": width,
            "rectified_height": height, "roi_inset_px": margin}


def rectangle_inside_mask(region: dict, mask: np.ndarray) -> bool:
    """Check every rasterized rectangle pixel, including concave paper boundaries."""
    quad = np.asarray(region["quad"], np.float32)
    height, width = mask.shape
    if (np.any(quad < 0) or np.any(quad[:, 0] > width - 1)
            or np.any(quad[:, 1] > height - 1)):
        return False
    polygon = np.rint(quad).astype(np.int32)
    x, y, w, h = cv2.boundingRect(polygon)
    coverage = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(coverage, polygon - (x, y), 1)
    return bool(np.all(mask[y:y + h, x:x + w][coverage.astype(bool)]))


def window_mask(region: dict, shape: tuple[int, int]) -> np.ndarray:
    """Restore a legacy contour-masked history record; not a measurement mask."""
    mask = np.zeros(shape, np.uint8)
    contours = [np.asarray(c, np.int32).reshape(-1, 1, 2) for c in region["window_contours"]]
    cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
    return mask


def _triplet_score(regions: tuple[tuple[dict, np.ndarray], ...]) -> float | None:
    bounds = [r for r, _ in regions]
    sizes = np.array([r["rectified_width"] * r["rectified_height"] for r in bounds], np.float64)
    if sizes.max() / sizes.min() > MAX_SIZE_RATIO:
        return None
    centers = np.array([np.mean(r["quad"], axis=0) for r in bounds])
    centered = centers - centers.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    alignment = float(np.max(np.abs(centered @ axes[1])) / np.sqrt(np.median(sizes)))
    spacing = np.diff(np.sort(centered @ axes[0]))
    if (alignment > MAX_ALIGNMENT_ERROR or spacing.min() < 1.25 * np.sqrt(np.median(sizes))
            or spacing.max() / spacing.min() > MAX_SPACING_RATIO):
        return None
    return (alignment + float(np.std(np.log(sizes)))
            + float(abs(spacing[0] - spacing[1]) / spacing.mean())
            + 0.1 * (1 - float(np.mean([r["rectangularity"] for r in bounds]))))


def detect_windows(gray: np.ndarray) -> WindowDetection:
    """Limit candidates to the paper interior, then validate a unique three-window layout."""
    paper = detect_paper_mask(gray)
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    binary[~paper.interior] = 0
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary[~paper.interior] = 0
    inside = paper.interior.astype(np.uint8)
    edge = paper.interior & ~cv2.erode(inside, kernel, borderType=cv2.BORDER_CONSTANT,
                                      borderValue=0).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []
    rejected = {"boundary": 0, "shape": 0, "rectangle_outside": 0}
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        component = labels[y:y + h, x:x + w] == index
        if np.any(component & edge[y:y + h, x:x + w]):
            rejected["boundary"] += 1
            continue
        region = oriented_region({"x": x, "y": y, "width": w, "height": h, "area": area}, component)
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
        rw, rh = region["rectified_width"], region["rectified_height"]
        region["rectangularity"] = float(np.count_nonzero(filled) / (rw * rh))
        if not 0.7 <= rw / rh <= 1.3 or region["rectangularity"] < MIN_RECTANGULARITY:
            rejected["shape"] += 1
            continue
        if not rectangle_inside_mask(region, paper.interior):
            rejected["rectangle_outside"] += 1
            continue
        region["window_contours"] = [(c.reshape(-1, 2) + (x, y)).tolist() for c in contours]
        candidates.append((region, component))
    candidates.sort(key=lambda item: (item[0]["x"], item[0]["y"]))
    if len(candidates) < 3:
        raise ValueError("공인인증 종이 내부 영역은 정확히 3개여야 합니다. "
                         f"검출된 영역: {len(candidates)}개 "
                         f"(경계 접촉 {rejected['boundary']}개, 회전 ROI 경계 초과 {rejected['rectangle_outside']}개)")
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"공인인증 종이 내부 후보가 너무 많습니다: {len(candidates)}개. 종이와 조명을 확인하세요.")
    ranked = []
    for triplet in combinations(candidates, 3):
        score = _triplet_score(triplet)
        if score is not None:
            ranked.append((score, triplet))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        raise ValueError(f"공인인증 종이 내부 후보 {len(candidates)}개의 크기·정렬·간격이 3점 배치와 맞지 않습니다.")
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < AMBIGUITY_GAP:
        raise ValueError(f"공인인증 종이 내부 후보 {len(candidates)}개의 P1~P3 선택이 모호합니다.")
    metadata = {"method": "paper_window_triplet", "paper": paper.metadata,
                "candidate_count": len(candidates), "selected_count": 3,
                "rejected": rejected, "layout_score": ranked[0][0],
                "limits": {"dark_threshold": THRESHOLD, "min_area": MIN_AREA, "max_area": MAX_AREA,
                           "min_rectangularity": MIN_RECTANGULARITY, "max_size_ratio": MAX_SIZE_RATIO,
                           "max_alignment_error": MAX_ALIGNMENT_ERROR, "max_spacing_ratio": MAX_SPACING_RATIO,
                           "ambiguity_gap": AMBIGUITY_GAP}}
    return WindowDetection(paper, ranked[0][1], metadata)


def detect_regions(gray: np.ndarray) -> tuple[tuple[dict, np.ndarray], ...]:
    """Compatibility entry point returning the three validated native-coordinate windows."""
    return detect_windows(gray).regions
