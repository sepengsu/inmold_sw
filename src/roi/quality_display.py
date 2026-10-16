"""Explain ROI decisions using the settings captured with that extraction."""

from math import isfinite
from .quality_profiles import ROI_LEVEL_LABELS


def roi_score_message(metadata: dict) -> str:
    score = metadata["quality_score"]
    threshold = metadata.get("quality_threshold")
    if threshold is None:
        return f"경계 품질 {score:.3f}"
    passed = isfinite(score) and score >= threshold
    state, comparison = ("통과", "≥") if passed else ("미달", "<")
    level = metadata.get("validation_level")
    suffix = f" · 검증 {ROI_LEVEL_LABELS[level]}" if level in ROI_LEVEL_LABELS else ""
    return f"점수 조건 {state}: {score:.3f} {comparison} 적용 기준 {threshold:.3f}{suffix}"


def roi_quality_message(metadata: dict) -> str:
    lines = [roi_score_message(metadata)]
    reasons = [reason for reason in metadata.get("needs_review_reason", "").split(";") if reason]
    if not reasons:
        return lines[0]
    limits = metadata.get("quality_limits", {})
    fits = metadata.get("boundary_fits", {})
    labels = {
        "LOW_BOUNDARY_SCORE": "경계 품질 점수 기준 미달",
        "LARGE_ANGLE_DIFFERENCE": "상·하단 경계 각도 차이 초과",
        "LARGE_HEIGHT_VARIATION": "위치에 따른 ROI 높이 변화 초과",
        "EXCESSIVE_TILT": "경계 기울기 허용 범위 초과",
        "INVALID_GEOMETRY": "유효하지 않은 경계 위치 또는 상·하단 순서",
        "IMPLAUSIBLE_HEIGHT": "직물 높이가 설정된 범위를 벗어남",
        "AMBIGUOUS_CANDIDATES": "서로 다른 경계 후보들의 점수가 비슷해 선택이 불확실함",
        "LOW_MULTI_EXPOSURE_CONSENSUS": "촬영 간 경계 일치 부족",
    }
    lines.append("자동 검사 보류 사유:")
    for reason in reasons:
        detail = labels.get(reason, reason)
        for side, name in (("top", "상단"), ("bottom", "하단")):
            if not reason.endswith("_" + side.upper()):
                continue
            fit = fits.get(side, {})
            if reason.startswith("RANSAC_FAILED_"):
                detail = f"{name} 경계 직선 피팅 실패"
            elif reason.startswith("LOW_INLIER_RATIO_"):
                ratio, minimum = fit.get("inlier_ratio"), limits.get("min_inlier_ratio")
                detail = (f"{name} 경계 일치 비율 {ratio:.1%} < 최소 {minimum:.1%}"
                          if ratio is not None and minimum is not None else f"{name} 경계 일치 비율 부족")
            elif reason.startswith("MISSING_THIRD_SUPPORT_"):
                minimum = limits.get("min_third_support")
                missing = [f"{label} {fit[key]:g}개" for key, label in
                           (("left_support", "왼쪽"), ("center_support", "중앙"), ("right_support", "오른쪽"))
                           if minimum is not None and fit.get(key) is not None and fit[key] < minimum]
                detail = (f"{name} 지지점 부족: {', '.join(missing)} / 각 구간 최소 {minimum:g}개"
                          if missing else f"{name} 좌·중·우 일부 구간의 경계 지지점 부족")
            else:
                prefixes = {
                    "LOW_X_COVERAGE_": "가로 방향 경계 지지 범위 부족",
                    "LONG_UNSUPPORTED_GAP_": "경계 근거가 끊기는 구간이 너무 김",
                    "HIGH_RANSAC_RMSE_": "경계 직선 오차 초과",
                    "HIGH_MEDIAN_RESIDUAL_": "경계 오차 중앙값 초과",
                    "OUTSIDE_CHECK_UNAVAILABLE_": "경계 바깥 영역 확인 불가",
                    "FABRIC_OUTSIDE_": "ROI 바깥에도 직물 무늬가 검출됨",
                }
                detail = next((f"{name} {text}" for prefix, text in prefixes.items() if reason.startswith(prefix)), reason)
        lines.append(f"- {detail} [{reason}]")
    return "\n".join(lines)
