"""Linked ROI quality settings, from relaxed to strict validation."""

from math import ceil, isclose


ROI_LEVEL_LABELS = {"low": "낮음 · 완화", "medium": "중간", "high": "높음 · 엄격"}
ROI_QUALITY_PROFILES = {
    "low": {
        "quality": {"min_score": .25, "min_inlier_ratio": .35, "min_x_coverage": .40,
                    "min_third_support": 0, "max_gap_ratio": .60, "max_rmse_px": 14.0,
                    "max_median_residual_px": 14.0, "max_tilt_deg": 25.0,
                    "max_angle_diff_deg": 6.0, "max_height_variation_px": 200.0,
                    "ambiguity_score_gap": .01, "ambiguity_boundary_gap_px": 50.0,
                    "min_consensus_ratio": .35},
        "ransac": {"thresholds": [12.0, 18.0, 24.0]},
    },
    "medium": {
        "quality": {"min_score": .40, "min_inlier_ratio": .55, "min_x_coverage": .70,
                    "min_third_support": 2, "max_gap_ratio": .25, "max_rmse_px": 8.0,
                    "max_median_residual_px": 8.0, "max_tilt_deg": 20.0,
                    "max_angle_diff_deg": 2.5, "max_height_variation_px": 100.0,
                    "ambiguity_score_gap": .02, "ambiguity_boundary_gap_px": 30.0,
                    "min_consensus_ratio": .50},
        "ransac": {"thresholds": [8.0, 12.0, 16.0]},
    },
    "high": {
        "quality": {"min_score": .55, "min_inlier_ratio": .70, "min_x_coverage": .85,
                    "min_third_support": 3, "max_gap_ratio": .15, "max_rmse_px": 5.0,
                    "max_median_residual_px": 5.0, "max_tilt_deg": 15.0,
                    "max_angle_diff_deg": 1.5, "max_height_variation_px": 60.0,
                    "ambiguity_score_gap": .03, "ambiguity_boundary_gap_px": 20.0,
                    "min_consensus_ratio": .70},
        "ransac": {"thresholds": [4.0, 6.0, 8.0]},
    },
}


def linked_roi_settings(score: float) -> dict:
    """Interpolate supporting gates with the score; retain the exact score value."""
    profiles = list(ROI_QUALITY_PROFILES.values())
    left, right = profiles[0], profiles[0]
    for lower, upper in zip(profiles, profiles[1:]):
        if lower["quality"]["min_score"] <= score <= upper["quality"]["min_score"]:
            left, right = lower, upper
            break
    else:
        if score >= profiles[-1]["quality"]["min_score"]:
            left = right = profiles[-1]
    start, end = left["quality"]["min_score"], right["quality"]["min_score"]
    fraction = (score - start) / (end - start) if end != start else 0.
    quality = {}
    for key, value in left["quality"].items():
        interpolated = value + (right["quality"][key] - value) * fraction
        quality[key] = max(0, ceil(interpolated - 1e-9)) if key == "min_third_support" else round(interpolated, 9)
    quality["min_score"] = score
    thresholds = [round(a + (b - a) * fraction, 9)
                  for a, b in zip(left["ransac"]["thresholds"], right["ransac"]["thresholds"])]
    return {"quality": quality, "ransac": {"thresholds": thresholds}}


def roi_level(config: dict) -> str | None:
    for name, profile in ROI_QUALITY_PROFILES.items():
        if (all(isclose(config["quality"].get(key, float("nan")), value, abs_tol=1e-9)
                for key, value in profile["quality"].items())
                and config["ransac"]["thresholds"] == profile["ransac"]["thresholds"]):
            return name
    return None


def roi_settings_summary(config: dict) -> str:
    quality = config["quality"]
    support = (f"구간별 지지점 ≥ {quality['min_third_support']}개" if quality["min_third_support"]
               else "좌·중·우 각 구간의 개별 지지점 의무 없음")
    return (f"점수 ≥ **{quality['min_score']:g}** · 경계 일치 ≥ **{quality['min_inlier_ratio']:.1%}** · "
            f"가로 지지 범위 ≥ **{quality['min_x_coverage']:.1%}** · 끊김 ≤ **{quality['max_gap_ratio']:.1%}**\n\n"
            f"{support} · 직선 오차 ≤ {quality['max_rmse_px']:g}px · 상·하단 각도 차이 ≤ {quality['max_angle_diff_deg']:g}°")
