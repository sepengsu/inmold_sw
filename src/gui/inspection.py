"""Basler RGB frame to the project's Exposure segmentation and UI results."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np

from src.model import load_checkpoint, predict_tiles
from src.postprocess import impregnation_pct, threshold_mask
from src.preprocess import load_config, preprocess_roi_array
from src.preprocess.auto_roi import RoiQualityError, draw_roi_boundaries
from src.preprocess._validation import integer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINT_WIDTH = 256
POINT_HEIGHT = 512
POINT_SPACING = 547
TILE_SIZE = 256


@dataclass(frozen=True, slots=True)
class InspectionResult:
    scores: tuple[float, float, float]
    average: float
    annotated_frame: np.ndarray
    crops: tuple[np.ndarray, np.ndarray, np.ndarray]
    roi_detection: dict | None = None
    regions: tuple[dict, ...] = ()
    roi: dict | None = None


def plan_measurement_regions(roi: dict, valid_mask: np.ndarray) -> tuple[dict, dict, dict]:
    """Place native 256x512 points on the ROI midline, 547 pixels apart.

    Return original-image coordinates. A tilted ROI supplies a separate vertical
    midpoint at each point's horizontal pixel center. Never resize, pad, or shift
    a measurement to make it fit; every measurement pixel must be inside the ROI.
    """
    x, y, width, height = (
        integer(roi[key], f"roi.{key}", 0 if key in ("x", "y") else 1)
        for key in ("x", "y", "width", "height")
    )
    if (not isinstance(valid_mask, np.ndarray) or valid_mask.dtype != np.bool_
            or valid_mask.shape != (height, width)):
        raise ValueError("ROI mask shape differs from ROI bounds")
    if width < 2 * POINT_SPACING + POINT_WIDTH or height < POINT_HEIGHT:
        raise ValueError("P1~P3 256×512 측정 영역을 배치하기에는 ROI 크기가 부족합니다. "
                         "최소 가로 1350px, 세로 512px이 필요합니다.")
    center_x = (width - POINT_WIDTH) // 2
    regions = []
    for index, offset in enumerate((-POINT_SPACING, 0, POINT_SPACING), start=1):
        left = center_x + offset
        # Even-width crops have a half-integer horizontal pixel center.
        center_columns = (left + POINT_WIDTH // 2 - 1, left + POINT_WIDTH // 2)
        midpoints = []
        for column in center_columns:
            rows = np.flatnonzero(valid_mask[:, column])
            if not rows.size:
                raise ValueError(f"P{index} 중심에 유효한 ROI가 없습니다.")
            midpoints.append((int(rows[0]) + int(rows[-1])) / 2)
        center_y = sum(midpoints) / len(midpoints)
        top = int(np.floor(center_y - (POINT_HEIGHT - 1) / 2 + 0.5))
        if (top < 0 or top + POINT_HEIGHT > height
                or not valid_mask[top:top + POINT_HEIGHT, left:left + POINT_WIDTH].all()):
            raise ValueError(f"P{index} 256×512 측정 영역이 ROI 경계를 벗어납니다. "
                             "촬영 위치 또는 ROI를 확인하세요.")
        regions.append({"x": x + left, "y": y + top,
                        "width": POINT_WIDTH, "height": POINT_HEIGHT})
    return tuple(regions)


def score_regions(exposures: tuple[np.ndarray, ...]) -> tuple[float, float, float]:
    """Score exactly three complete native measurement masks (131,072 pixels each)."""
    if len(exposures) != 3 or any(
        not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
        or mask.shape != (POINT_HEIGHT, POINT_WIDTH) for mask in exposures
    ):
        raise ValueError("P1~P3 마스크는 각각 256×512 크기의 bool 배열이어야 합니다.")
    return tuple(impregnation_pct(mask) for mask in exposures)


def annotate_frame(
    frame_rgb: np.ndarray, exposures: tuple[np.ndarray, ...], regions: tuple[dict, ...],
    roi_detection: dict | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Overlay only measured Exposure pixels, with boxes and crops at the same coordinates."""
    annotated = frame_rgb.copy()
    red = np.array([235, 64, 64], dtype=np.uint8)
    for mask, region in zip(exposures, regions, strict=True):
        x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
        view = annotated[y:y + height, x:x + width]
        view[mask] = (view[mask].astype(np.uint16) + red.astype(np.uint16)) // 2
    if roi_detection is not None:
        annotated = draw_roi_boundaries(annotated, roi_detection)
    colors = ((255, 220, 0), (32, 215, 232), (234, 85, 223))
    crops = []
    for index, (region, color) in enumerate(zip(regions, colors, strict=True), start=1):
        x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
        cv2.rectangle(annotated, (x, y), (x + width - 1, y + height - 1), color, 2)
        cv2.putText(annotated, f"P{index}", (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        crops.append(annotated[y:y + height, x:x + width].copy())
    return annotated, tuple(crops)


class InspectionEngine:
    """Load the checkpoint once and inspect copies of live camera frames."""

    def __init__(
        self, config_path: Path = PROJECT_ROOT / "config" / "preprocess.yaml",
        checkpoint_path: Path = PROJECT_ROOT / "model" / "best.pt",
        *, device: str = "cpu", batch_size: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.checkpoint_path = Path(checkpoint_path)
        self.config = load_config(config_path)
        self.last_roi_detection: dict | None = None
        self.model, self.checkpoint = load_checkpoint(checkpoint_path, device=device)
        with self.checkpoint_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.model_info = {**self.checkpoint, "checkpoint": str(self.checkpoint_path.resolve()),
                           "sha256": digest}

    def inspect(self, frame_rgb: np.ndarray) -> InspectionResult:
        if (not isinstance(frame_rgb, np.ndarray) or frame_rgb.dtype != np.uint8
                or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3):
            raise ValueError("카메라 프레임은 uint8 RGB 배열이어야 합니다.")
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        try:
            prepared = preprocess_roi_array(gray, self.config)
        except RoiQualityError as error:
            self.last_roi_detection = error.result.metadata
            raise
        self.last_roi_detection = prepared["metadata"].get("roi_detection")
        roi = prepared["metadata"]["roi"]
        regions = plan_measurement_regions(roi, prepared["valid_mask"])
        patches = []
        for region in regions:
            x, y = region["x"] - roi["x"], region["y"] - roi["y"]
            for offset_y in (0, TILE_SIZE):
                patches.append(prepared["clahe"][y + offset_y:y + offset_y + TILE_SIZE,
                                                 x:x + TILE_SIZE].copy())
        probabilities = predict_tiles(self.model, patches, batch_size=self.batch_size)
        if len(probabilities) != 6 or any(p.shape != (TILE_SIZE, TILE_SIZE) for p in probabilities):
            raise ValueError("P1~P3 추론에는 256×256 예측 마스크 6장이 필요합니다.")
        exposures = tuple(threshold_mask(np.concatenate(probabilities[index:index + 2], axis=0),
                                        self.checkpoint["threshold"]) for index in (0, 2, 4))
        scores = score_regions(exposures)
        annotated, crops = annotate_frame(frame_rgb, exposures, regions, self.last_roi_detection)
        return InspectionResult(scores, float(np.mean(scores)), annotated, crops,
                                self.last_roi_detection, regions, dict(roi))

    def inspect_certification(self, frame_rgb: np.ndarray):
        from src.PCE import inspect_frame

        self.last_roi_detection = None
        return inspect_frame(frame_rgb, self.model, self.checkpoint["threshold"], self.config)
