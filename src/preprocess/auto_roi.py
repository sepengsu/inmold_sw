"""Single-frame adapter for the upstream Gabor/DP/RANSAC ROI detector."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ._validation import grayscale
from src.roi.quality_display import roi_quality_message
from src.roi.quality_profiles import roi_level


UPSTREAM_COMMIT = "dcaafcd5a03903699161b2ded5d2aa1add1c8b8c"


@dataclass(frozen=True, slots=True)
class AutoRoiResult:
    mask: np.ndarray
    metadata: dict

    @property
    def accepted(self) -> bool:
        return (self.metadata["roi_detected"]
                and self.metadata["quality_status"] in {"PASS", "RECOVERED"})

    @property
    def bounds(self) -> dict[str, int]:
        ys, xs = np.nonzero(self.mask)
        if not len(ys):
            raise ValueError("검출된 ROI 픽셀이 없습니다.")
        return {"x": int(xs.min()), "y": int(ys.min()),
                "width": int(xs.max() - xs.min() + 1), "height": int(ys.max() - ys.min() + 1)}


class RoiQualityError(ValueError):
    def __init__(self, result: AutoRoiResult):
        self.result = result
        super().__init__(
            f"자동 ROI 검토 필요 ({result.metadata['quality_status']})\n"
            f"{roi_quality_message(result.metadata)}"
        )


def detect_auto_roi(
    image: np.ndarray, config_path: str | Path | None = None, *, require_accepted: bool = True,
) -> AutoRoiResult:
    """Detect at native resolution; never turn a review fallback into acceptance."""
    grayscale(image)
    from src.roi import auto_roi_local_std as core
    from src.roi import roi_boundary as boundary
    from src.roi import roi_boundary_quality as quality

    config = quality.load_config(Path(config_path) if config_path is not None else None)
    feature = boundary.features(image, config)
    candidates, _ = boundary.detect_candidates(feature, config)
    args = SimpleNamespace(mode="gabor_boundary", margin=0, boundary_config=config)
    # A live frame is one exposure; do not fabricate multi-exposure consensus.
    result = boundary.finalize(candidates, feature, config, [], 1, args)
    mask = core.roi_pixel_mask(image.shape, result)
    metadata = {
        "method": core.BOUNDARY_DETECTOR, "upstream_commit": UPSTREAM_COMMIT,
        "source_shape": list(image.shape), "roi_detected": bool(result["roi_detected"]),
        "quality_status": result["quality_status"],
        "quality_score": float(result["roi_quality"]),
        "quality_threshold": float(config["quality"]["min_score"]),
        "validation_level": roi_level(config),
        "quality_limits": dict(config["quality"]),
        "boundary_fits": {
            side: {
                key: float(result[side + "_fit"][key]) if result[side + "_fit"].get(key) is not None else None
                for key in ("inlier_ratio", "left_support", "center_support", "right_support")
            }
            for side in ("top", "bottom")
        },
        "needs_review_reason": result["needs_review_reason"] or "",
        "top_y": result["top_y"].tolist(), "bottom_y_exclusive": result["bottom_y"].tolist(),
        "top_line": dict(result["top_line"]), "bottom_line": dict(result["bottom_line"]),
        "margin_top": result["roi_margin_top"], "margin_bottom": result["roi_margin_bottom"],
        "mask_rounding": result["mask_rounding"],
    }
    detected = AutoRoiResult(mask=mask, metadata=metadata)
    if require_accepted and not detected.accepted:
        raise RoiQualityError(detected)
    return detected


def draw_roi_boundaries(image_rgb: np.ndarray, metadata: dict) -> np.ndarray:
    """Draw native-pixel mask edges without resampling or changing the input."""
    output = image_rgb.copy()
    height, width = output.shape[:2]
    if metadata["source_shape"] != [height, width]:
        raise ValueError("ROI 크기가 현재 카메라 영상과 다릅니다. 다시 추출하세요.")
    color = (40, 220, 100) if metadata["quality_status"] in {"PASS", "RECOVERED"} else (255, 180, 0)
    xs = np.arange(width, dtype=np.int32)
    for key, delta in (("top_y", 0), ("bottom_y_exclusive", -1)):
        ys = np.clip(np.asarray(metadata[key], dtype=np.int32) + delta, 0, height - 1)
        points = np.stack((xs, ys), axis=1).reshape(-1, 1, 2)
        cv2.polylines(output, [points], False, color, 2, cv2.LINE_AA)
    return output
