"""Core ROI extraction, geometry, masking, and batch export implementation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.signal import find_peaks
from skimage.measure import LineModelND, ransac


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MODES = ("std", "fft", "gabor", "fft+gabor")
MODE_LABELS = {"std": "STD", "fft": "FFT", "gabor": "Gabor", "fft+gabor": "FFT+Gabor"}
DETECTOR = "boundary_evidence_center_dp_quality_ransac_v3"
AUTO_DETECTOR = "group_fusion_candidates_consensus_registration_v4_1"
BOUNDARY_DETECTOR = "gabor_boundary_quality_v5"
BOUNDARY_FIELDS = (
    "scene_id", "image_path", "exposure_us", "reference_image", "quality_status",
    "angle_diff_deg", "roi_height_left", "roi_height_center", "roi_height_right", "height_variation_px",
    "top_longest_gap", "bottom_longest_gap", "top_ransac_rmse", "bottom_ransac_rmse",
    "best_candidate_score", "second_candidate_score", "candidate_score_gap", "candidate_boundary_gap_px",
    "ambiguous_candidate_score", "ambiguous_boundary_gap_px", "outside_texture_top", "outside_texture_bottom",
    "consensus_total", "consensus_ratio", "was_recovered", "recovery_method", "needs_review_reason",
    "manual_override", "roi_margin_top", "roi_margin_bottom", "mask_rounding", "quality_details_json",
    "automatic_prediction_json", "debug_path",
    "gt_top_rmse_px", "gt_bottom_rmse_px", "gt_top_max_error_px", "gt_bottom_max_error_px",
    "gt_top_p95_error_px", "gt_bottom_p95_error_px", "gt_under_crop_pixels", "gt_over_crop_pixels",
    "gt_under_crop_fraction", "gt_over_crop_fraction", "gt_mask_iou",
)

CSV_FIELDS = (
    "source", "scene", "camera", "exposure", "mode", "roi_source", "reference_exposure",
    "is_reference", "roi_reused", "status", "reason", "found", "needs_review", "output_written",
    "output_path", "image_output", "input_height", "input_width", "channels", "dtype",
    "output_height", "output_width", "output_origin_y", "margin", "y0", "y1_exclusive",
    "roi_area_ratio", "height_left", "height_right", "height_ratio_mean",
    "roi_gray_mean", "roi_gray_std", "roi_zero_fraction", "roi_saturated_fraction",
    "selected_pair_count", "candidate_column_count", "empty_candidate_columns", "pair_score_mean",
    "anchor_x", "height_mean_px", "height_std_px", "height_change_ratio", "pair_inlier_ratio",
    "gt_split", "gt_auto_status", "gt_auto_found", "gt_top_mae_px", "gt_bottom_mae_px",
    "auto_accepted", "export_eligible", "roi_detected", "roi_quality", "roi_candidate_count", "selection_method",
    "fallback_used", "quality_components_json", "candidates_json", "fusion_sources_json",
    "fusion_count", "consensus_count", "consensus_spread_px", "registration_source",
    "registration_correlation", "registration_transform_json", "output_warning",
) + tuple(
    f"{side}_{field}" for side in ("top", "bottom") for field in (
        "valid", "point_count", "inlier_count", "inlier_ratio", "span_ratio", "slope", "angle_deg",
        "intercept", "y_left", "y_right", "rmse_px", "tolerance_px", "attempt_count", "reacquired",
        "rejection_reason", "initial_rejection_reason", "points_json", "inliers_json",
        "x_coverage", "max_gap_ratio", "left_support", "center_support", "right_support",
        "median_residual_px",
    )
) + ("homography_json", "detector", "run_started_utc", "parameters_json") + BOUNDARY_FIELDS


class CsvReport:
    def __init__(self, args):
        self.args = args
        self.file = None
        self.started = datetime.now(timezone.utc).isoformat()
        self.parameters = json.dumps({k: str(v) if isinstance(v, Path) else v
                                      for k, v in vars(args).items()}, ensure_ascii=False, sort_keys=True)

    def __enter__(self):
        if not self.args.dry_run:
            if self.args.output.exists() and any(self.args.output.iterdir()):
                raise SystemExit("Output directory is not empty. Use a new --output folder to avoid stale results.")
            self.args.output.mkdir(parents=True, exist_ok=True)
            self.file = (self.args.output / "roi_results.csv").open("x", encoding="utf-8-sig", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
            self.writer.writeheader()
            self.file.flush()
        return self

    def write(self, row):
        if self.file is not None:
            detector = (BOUNDARY_DETECTOR if self.args.mode == "gabor_boundary" else
                        AUTO_DETECTOR if self.args.mode == "auto" else DETECTOR)
            row = {**row, "detector": detector,
                   "run_started_utc": self.started,
                   "parameters_json": self.parameters}
            self.writer.writerow({k: int(v) if isinstance(v, (bool, np.bool_)) else v for k, v in row.items()})
            self.file.flush()

    def __exit__(self, *_):
        if self.file is not None:
            self.file.close()


def result_identity(image_path: Path, mode: str, args, reference_path: Path | None = None) -> dict:
    relative = image_path.relative_to(args.source)
    reference = reference_path.relative_to(args.source) if reference_path is not None else None
    return {
        "source": relative.as_posix(), "scene": relative.parent.parent.as_posix(),
        "camera": image_path.stem.split("_", 1)[0], "exposure": image_path.parent.name, "mode": mode,
        "roi_source": reference.as_posix() if reference is not None else "",
        "reference_exposure": reference.parent.name if reference is not None else args.reference_exposure,
        "is_reference": reference == relative, "roi_reused": reference is not None and reference != relative,
        "image_output": args.image_output, "output_written": False, "output_path": "",
    }


def record_skipped(report: CsvReport, path: Path, modes: tuple, args, reason: str,
                   reference_path: Path | None = None, image: np.ndarray | None = None) -> None:
    for mode in modes:
        row = result_identity(path, mode, args, reference_path)
        row.update(status="skipped", reason=reason, needs_review=True)
        if image is not None:
            row.update(input_height=image.shape[0], input_width=image.shape[1], dtype=str(image.dtype))
        report.write(row)


def parse_extensions(value: str) -> set[str]:
    extensions = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else f".{item}")
    return extensions


def imread(path: Path) -> np.ndarray | None:
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def iter_images(
    source: Path,
    output: Path,
    extensions: set[str],
    prefix: str | None,
    exposure: str | None = "10000us",
):
    output = output.resolve()
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        if prefix and not path.name.startswith(prefix):
            continue
        if exposure is not None and path.parent.name.casefold() != exposure.casefold():
            continue
        if exposure is None and not re.fullmatch(r"\d+us", path.parent.name, re.IGNORECASE):
            continue
        try:
            path.resolve().relative_to(output)
            continue
        except ValueError:
            pass
        yield path


def reference_jobs(images: list[Path], exposure: str) -> tuple[list, list]:
    groups = defaultdict(list)
    for path in images:
        # The parent of the exposure folder identifies the scene and direction.
        # Keep camera prefixes separate, even when --prefix includes both cameras.
        camera = path.stem.split("_", 1)[0].casefold()
        groups[(path.parent.parent, camera)].append(path)
    jobs, issues = [], []
    for paths in groups.values():
        references = [p for p in paths if p.parent.name.casefold() == exposure.casefold()]
        by_name = {p.name.casefold(): p for p in references}
        multiple_frames = max(Counter(p.parent for p in paths).values()) > 1
        assignments = defaultdict(list)
        for target in paths:
            reference = by_name.get(target.name.casefold())
            if reference is None and len(references) == 1 and not multiple_frames:
                reference = references[0]
            if reference is None:
                reason = "missing_reference" if not references else "ambiguous_reference"
                issues.append({"source": target, "reason": reason})
            else:
                assignments[reference].append(target)
        jobs.extend((reference, sorted(targets)) for reference, targets in assignments.items())
    return sorted(jobs, key=lambda job: job[0]), issues


def odd_at_most(value: int, limit: int) -> int:
    value = max(1, int(value))
    limit = max(1, int(limit))
    value = min(value, limit)
    if value % 2 == 0:
        value -= 1
    return max(1, value)


def gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(image, conversion)
    return gray.astype(np.float32)


def std_row_score(g: np.ndarray, kernel: int, x0: int, x1: int) -> np.ndarray:
    std_k = odd_at_most(kernel, min(g.shape))
    mean = cv2.boxFilter(
        g,
        -1,
        (std_k, std_k),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    mean_sq = cv2.boxFilter(
        g * g,
        -1,
        (std_k, std_k),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    return local_std[:, x0:x1].mean(axis=1)


def fft_row_score(
    g: np.ndarray,
    x0: int,
    x1: int,
    window_size: int,
    stride: int,
    min_frequency: float,
    max_frequency: float,
) -> np.ndarray:
    # Local 2-D spectra retain Y position; a single global spectrum would not.
    h = g.shape[0]
    band = g[:, x0:x1]
    size = min(window_size, *band.shape)
    if size < 4:
        return np.zeros(h, dtype=np.float32)
    step = min(stride, size)
    half = size // 2
    padded = cv2.copyMakeBorder(
        band, half, half, half, half, cv2.BORDER_REFLECT
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    ys = np.unique(np.append(np.arange(0, h, step), h - 1))
    xs = np.unique(np.append(np.arange(0, band.shape[1], step), band.shape[1] - 1))
    hann = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    frequency = np.hypot(fy, fx)
    bandpass = (frequency >= min_frequency) & (frequency <= max_frequency)
    # Restore the negative-X half of the power spectrum, excluding DC/Nyquist.
    weights = np.full(size // 2 + 1, 2.0, dtype=np.float32)
    weights[0] = 1.0
    if size % 2 == 0:
        weights[-1] = 1.0
    weighted_band = bandpass * weights[None, :]
    normalizer = size * size * float(np.square(hann).sum())
    scores = []
    for y in ys:
        patches = windows[y, xs].copy()
        patches -= patches.mean(axis=(-2, -1), keepdims=True)
        spectrum = np.fft.rfft2(patches * hann, axes=(-2, -1))
        power = spectrum.real**2 + spectrum.imag**2
        rms = np.sqrt((power * weighted_band).sum(axis=(-2, -1)) / normalizer)
        scores.append(float(rms.mean()))
    return np.interp(np.arange(h), ys, scores).astype(np.float32)


@lru_cache(maxsize=16)
def gabor_kernels(size: int, wavelengths: tuple[float, ...]) -> tuple:
    pairs = []
    for wavelength in wavelengths:
        for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            pair = []
            for phase in (0.0, np.pi / 2):
                kernel = cv2.getGaborKernel(
                    (size, size), 0.4 * wavelength, theta, wavelength,
                    0.5, phase, ktype=cv2.CV_32F,
                )
                kernel -= kernel.mean()
                kernel /= max(float(np.abs(kernel).sum()), 1e-12)
                pair.append(kernel)
            pairs.append(tuple(pair))
    return tuple(pairs)


def gabor_row_score(
    g: np.ndarray, x0: int, x1: int, kernel: int, wavelengths: tuple[float, ...]
) -> np.ndarray:
    size = odd_at_most(kernel, min(g.shape))
    # Include neighboring pixels so the central band's edges do not become texture.
    left, right = max(0, x0 - size // 2), min(g.shape[1], x1 + size // 2)
    band = g[:, left:right]
    score = np.zeros(g.shape[0], dtype=np.float32)
    pairs = gabor_kernels(size, wavelengths)
    for even, odd in pairs:
        real = cv2.filter2D(band, -1, even, borderType=cv2.BORDER_REFLECT)
        imaginary = cv2.filter2D(band, -1, odd, borderType=cv2.BORDER_REFLECT)
        energy = cv2.magnitude(real, imaginary)
        score += energy[:, x0 - left:x1 - left].mean(axis=1)
    return score / len(pairs)


def normalize_score(score: np.ndarray) -> np.ndarray:
    low, high = float(score.min()), float(score.max())
    if high - low <= max(1e-6, max(abs(low), abs(high)) * 1e-6):
        return np.zeros_like(score, dtype=np.float32)
    return ((score - low) / (high - low)).astype(np.float32)


def texture_row_scores(
    image: np.ndarray, modes: tuple[str, ...], args,
    x_bounds: tuple[int, int] | None = None,
) -> tuple:
    g = gray_float(image)
    h, w = g.shape
    center_x_ratio = args.center_x_ratio
    trim = max(0.0, min(0.49, (1.0 - center_x_ratio) / 2.0))
    x0 = int(w * trim)
    x1 = int(w * (1.0 - trim))
    if x1 <= x0:
        x0, x1 = 0, w
    if x_bounds is not None:
        x0, x1 = x_bounds

    required = set(modes)
    if "fft+gabor" in required:
        required.update(("fft", "gabor"))
    scores = {}
    flat = float(g.max() - g.min()) == 0.0
    for mode in MODES[:3]:
        if mode not in required:
            continue
        if flat:
            score = np.zeros(h, dtype=np.float32)
        elif mode == "std":
            score = std_row_score(g, args.std_kernel, x0, x1)
        elif mode == "fft":
            score = fft_row_score(
                g, x0, x1, args.fft_window, args.fft_stride,
                args.fft_min_frequency, args.fft_max_frequency,
            )
        else:
            score = gabor_row_score(
                g, x0, x1, args.gabor_kernel, tuple(args.gabor_wavelengths)
            )
        smooth_k = odd_at_most(args.smooth_kernel, h)
        if smooth_k > 1:
            score = cv2.GaussianBlur(score.reshape(-1, 1), (1, smooth_k), 0).ravel()
        scores[mode] = score
    if "fft+gabor" in modes:
        scores["fft+gabor"] = (
            normalize_score(scores["fft"]) + normalize_score(scores["gabor"])
        ) * 0.5
    return {mode: scores[mode] for mode in modes}, x0, x1


def boundary_evidence(row_score: np.ndarray, brightness_edge: np.ndarray | None = None,
                      texture_gap: int = 15, texture_window: int = 60,
                      weights: tuple = (0.45, 0.15, 0.40)) -> dict:
    profile = normalize_score(row_score)
    h = len(profile)
    gradient = np.gradient(profile) if h > 1 else np.zeros(h)
    gradient /= max(float(np.max(np.abs(gradient))), 1e-8)
    gap = min(texture_gap, max(1, h // 20))
    window = min(texture_window, max(gap + 1, h // 6))
    ys = np.arange(h)
    cumulative = np.r_[0.0, np.cumsum(profile, dtype=np.float64)]

    def average(start, end):
        start, end = np.clip(start, 0, h), np.clip(end, 0, h)
        return (cumulative[end] - cumulative[start]) / np.maximum(end - start, 1)

    contrast = average(ys + gap, ys + window) - average(ys - window, ys - gap)
    # Both windows need evidence. Image borders cannot manufacture contrast from padding.
    contrast[(ys < gap + 1) | (ys + gap >= h)] = 0
    brightness = np.zeros(h) if brightness_edge is None else normalize_score(brightness_edge)
    wg, wb, wt = np.asarray(weights) / sum(weights)
    result = {"gradient": gradient, "contrast": contrast, "brightness": brightness}
    for side, sign in (("top", 1), ("bottom", -1)):
        texture = sign * gradient
        difference = sign * contrast
        evidence = wg * np.maximum(texture, 0) + wb * brightness + wt * np.maximum(difference, 0)
        # Brightness alone, or a strong edge with texture on the wrong side, is insufficient.
        evidence[(texture < 0.025) | (difference < -0.02)] = 0
        result[side] = evidence
    return result


def boundary_pair_candidates(
    row_score: np.ndarray, min_height_ratio: float, max_height_ratio: float,
    smooth_kernel: int = 51, brightness_edge: np.ndarray | None = None,
    texture_gap: int = 15, texture_window: int = 60, weights: tuple = (0.45, 0.15, 0.40),
) -> list[dict]:
    h = len(row_score)
    if h < 5:
        return []
    profile = normalize_score(row_score)
    evidence = boundary_evidence(row_score, brightness_edge, texture_gap, texture_window, weights)
    distance = max(2, min(smooth_kernel // 2, h // 40))

    def peaks(values):
        locations, properties = find_peaks(values, height=0.05, prominence=0.03, distance=distance)
        order = np.argsort(properties["peak_heights"])[-12:]
        return sorted(int(locations[i]) for i in order)

    tops, bottoms = peaks(evidence["top"]), peaks(evidence["bottom"])
    min_height = max(2, int(np.ceil(h * min_height_ratio)))
    max_height = int(np.floor(h * max_height_ratio))
    gap = max(1, min(smooth_kernel // 2, min_height // 8))
    outside_width = max(4, int(round(h * 0.04)))
    cumulative = np.concatenate(([0.0], np.cumsum(profile, dtype=np.float64)))

    def average(start, end):
        start, end = max(0, start), min(h, end)
        return float((cumulative[end] - cumulative[start]) / (end - start)) if end > start else None

    pairs = []
    expected = (min_height_ratio + max_height_ratio) / 2
    for top in tops:
        for bottom in bottoms:
            height = bottom - top
            if not min_height <= height <= max_height:
                continue
            inside = average(top + gap, bottom - gap)
            outside = [v for v in (average(top - gap - outside_width, top - gap),
                                   average(bottom + gap, bottom + gap + outside_width)) if v is not None]
            if inside is None or not outside:
                continue
            edge = float(evidence["top"][top] + evidence["bottom"][bottom])
            background = float(np.mean(outside))
            height_penalty = abs(height / h - expected) / max(max_height_ratio - min_height_ratio, 1e-6)
            score = edge + 0.5 * (inside - background) - 0.15 * height_penalty
            if score <= 0.15:
                continue
            pairs.append({"y0": top, "y1": bottom, "score": float(score),
                          "edge_score": edge, "inside_score": inside,
                          "outside_score": background, "height_penalty": float(height_penalty),
                          "top_contrast": float(evidence["contrast"][top]),
                          "bottom_contrast": float(-evidence["contrast"][bottom])})
    return sorted(pairs, key=lambda pair: pair["score"], reverse=True)[:48]


def pair_dp_states(valid: list[dict], height: int, continuity_weight: float,
                   slope_weight: float, spacing: float, height_weight: float | None = None) -> tuple:
    positions = [np.array([[p["y0"], p["y1"]] for p in c["candidates"]]) for c in valid]
    scores = [np.array([p["score"] for p in c["candidates"]]) for c in valid]
    predecessors = []

    def transition(i):
        before, after = positions[i - 1], positions[i]
        movement = np.abs(before[:, None, :] - after[None, :, :]).sum(axis=2)
        heights_before = before[:, 1] - before[:, 0]
        heights_after = after[:, 1] - after[:, 0]
        height_change = np.abs(heights_before[:, None] - heights_after[None, :])
        gap = max(1.0, abs(valid[i]["x"] - valid[i - 1]["x"]) / spacing)
        weight = .5 * continuity_weight if height_weight is None else height_weight
        return (continuity_weight * movement + weight * height_change) / (height * gap)

    # Keep two successive choices as the DP state so constant tilt has no curvature penalty.
    costs = -scores[0][:, None] + transition(1) - scores[1][None, :]
    for i in range(2, len(valid)):
        previous_slope = (positions[i - 1][None, :, :] - positions[i - 2][:, None, :]) / (
            valid[i - 1]["x"] - valid[i - 2]["x"])
        next_slope = (positions[i][None, :, :] - positions[i - 1][:, None, :]) / (
            valid[i]["x"] - valid[i - 1]["x"])
        curvature = np.abs(previous_slope[:, :, None, :] - next_slope[None, :, :, :]).sum(axis=3)
        options = costs[:, :, None] + slope_weight * spacing / height * curvature
        links = np.argmin(options, axis=0)
        costs = np.min(options, axis=0) + transition(i) - scores[i][None, :]
        predecessors.append(links)
    return costs, predecessors


def trace_pair_path(valid: list[dict], predecessors: list, last: tuple) -> list[dict]:
    indices = [0] * len(valid)
    indices[-2], indices[-1] = last
    for i in range(len(valid) - 1, 1, -1):
        indices[i - 2] = int(predecessors[i - 2][indices[i - 1], indices[i]])
    return [{"x": column["x"], **column["candidates"][index]} for column, index in zip(valid, indices)]


def select_pair_path(
    columns: list[dict], height: int, continuity_weight: float, slope_weight: float = 12.0,
) -> list[dict]:
    valid = sorted((c for c in columns if c["candidates"]), key=lambda c: c["x"])
    if not valid:
        return []
    if len(valid) == 1:
        return [{"x": valid[0]["x"], **max(valid[0]["candidates"], key=lambda p: p["score"]),
                 "anchor_x": valid[0]["x"]}]
    spacing = max(1.0, float(np.median(np.diff(sorted(c["x"] for c in columns)))))
    if len(valid) == 2:
        costs, links = pair_dp_states(valid, height, continuity_weight, slope_weight, spacing)
        return trace_pair_path(valid, links, np.unravel_index(np.argmin(costs), costs.shape))

    left_x, right_x = min(c["x"] for c in columns), max(c["x"] for c in columns)
    central = [i for i in range(1, len(valid) - 1)
               if left_x + (right_x - left_x) / 3 <= valid[i]["x"] <= right_x - (right_x - left_x) / 3]
    if not central:
        central = [min(range(1, len(valid) - 1), key=lambda i: abs(valid[i]["x"] - (left_x + right_x) / 2))]

    def confidence(i):
        scores = sorted((p["score"] for p in valid[i]["candidates"]), reverse=True)
        return scores[0] + (scores[0] - scores[1] if len(scores) > 1 else scores[0])

    anchor = max(central, key=confidence)
    left, right = valid[:anchor + 1], list(reversed(valid[anchor:]))
    lc, ll = pair_dp_states(left, height, continuity_weight, slope_weight, spacing)
    rc, rl = pair_dp_states(right, height, continuity_weight, slope_weight, spacing)
    positions = lambda col: np.array([[p["y0"], p["y1"]] for p in col["candidates"]])
    center = positions(valid[anchor])
    ls = (center[None, :, :] - positions(left[-2])[:, None, :]) / (left[-1]["x"] - left[-2]["x"])
    rs = (positions(right[-2])[None, :, :] - center[:, None, :]) / (right[-2]["x"] - right[-1]["x"])
    curvature = np.abs(ls[:, :, None, :] - rs[None, :, :, :]).sum(axis=3)
    # Combine both DP messages at the anchor, then trace outward. Retain every
    # anchor candidate and its neighboring slope states, avoiding a greedy seed.
    scores = np.array([p["score"] for p in valid[anchor]["candidates"]])
    costs = lc[:, :, None] + rc.T[None, :, :] + scores[None, :, None]
    costs += slope_weight * spacing / height * curvature
    li, ai, ri = np.unravel_index(np.argmin(costs), costs.shape)
    path = trace_pair_path(left, ll, (li, ai)) + list(reversed(trace_pair_path(right, rl, (ri, ai))))[1:]
    return [{**p, "anchor_x": valid[anchor]["x"]} for p in path]


def detect_roi_y(
    row_score: np.ndarray, margin: int, min_height_ratio: float,
    x0: int, x1: int, reference: tuple[int, int] | None = None,
    max_height_ratio: float = 0.55, smooth_kernel: int = 51,
) -> dict:
    h = len(row_score)
    pairs = boundary_pair_candidates(row_score, min_height_ratio, max_height_ratio, smooth_kernel)
    if reference is not None:
        ref_start, ref_end = reference
        pairs = [pair for pair in pairs if min(pair["y1"], ref_end) - max(pair["y0"], ref_start)
                 >= 0.5 * (ref_end - ref_start)]
    found = bool(pairs)
    start, end = (pairs[0]["y0"], pairs[0]["y1"]) if found else (0, 0)
    y0 = max(0, start - margin) if found else 0
    y1 = min(h, end + margin) if found else 0
    mask = np.zeros(h, dtype=bool)
    mask[y0:y1] = True
    return {
        "y0": y0,
        "y1": y1,
        "found": found,
        "row_score": row_score,
        "score_u8": (normalize_score(row_score) * 255).astype(np.uint8),
        "mask": mask,
        "threshold": None,
        "pair_candidates": pairs,
        "detector": DETECTOR,
        "center_x0": x0,
        "center_x1": x1,
    }


def fit_boundary_line(
    points: list[tuple[float, float]], min_span: float, residual_threshold: float = 8.0,
    x_bounds: tuple[float, float] | None = None, max_rmse: float | None = None,
    min_inlier_ratio: float = 0.55, max_gap_ratio: float = 0.25,
    max_median_residual: float = 8.0,
    min_third_support: int = 1,
) -> dict:
    result = {"valid": False, "points": [[float(x), float(y)] for x, y in points],
              "inliers": [False] * len(points), "residual_threshold": float(residual_threshold),
              "rejection_reason": "insufficient_points"}
    if len(points) < 4:
        return result
    xy = np.asarray(points, dtype=np.float32)

    model, inliers = ransac(
        xy, LineModelND, min_samples=2, residual_threshold=residual_threshold,
        is_data_valid=lambda sample: float(np.ptp(sample[:, 0])) > 1e-6,
        max_trials=500, stop_probability=0.999, rng=42,
    )
    if model is None or inliers is None:
        result["rejection_reason"] = "no_model"
        return result
    result["inliers"] = inliers.tolist()
    result["inlier_count"] = int(inliers.sum())
    xs = xy[inliers, 0]
    result["inlier_span"] = float(np.ptp(xs)) if len(xs) else 0.0
    result["inlier_ratio"] = float(inliers.mean())
    if x_bounds is not None:
        left, right = x_bounds
        width = max(1.0, right - left)
        thirds = np.clip(((xs - left) / width * 3).astype(int), 0, 2)
        result.update(x_coverage=result["inlier_span"] / width,
                      max_gap_ratio=float(np.diff(np.r_[left, np.sort(xs), right]).max()) / width,
                      **{f"{side}_support": int(np.sum(thirds == i))
                         for i, side in enumerate(("left", "center", "right"))})
    endpoints = model.predict_y(np.array([0.0, 1.0]))
    if not np.isfinite(endpoints).all():
        result["rejection_reason"] = "nonfinite_line"
        return result
    line = float(endpoints[1] - endpoints[0]), float(endpoints[0])
    error = xy[inliers, 1] - (line[0] * xy[inliers, 0] + line[1])
    rmse = float(np.sqrt(np.mean(error**2)))
    median = float(np.median(np.abs(error)))
    result.update(slope=line[0], intercept=line[1], rmse=rmse,
                  median_residual_px=median, method="ransac")
    reason = None
    if int(inliers.sum()) < max(4, int(np.ceil(len(points) * min_inlier_ratio))):
        reason = "insufficient_inliers"
    elif result["inlier_span"] < min_span:
        reason = "insufficient_x_span"
    elif x_bounds is not None and any(result[f"{side}_support"] < min_third_support for side in ("left", "center", "right")):
        reason = "missing_third_support"
    elif x_bounds is not None and result["max_gap_ratio"] > max_gap_ratio:
        reason = "excessive_unsupported_gap"
    elif rmse > (residual_threshold if max_rmse is None else max_rmse):
        reason = "excessive_rmse"
    elif median > max_median_residual:
        reason = "excessive_median_residual"
    result.update(valid=reason is None, rejection_reason=reason)
    return result


def fit_boundary_with_retries(points: list, bounds: tuple, args) -> dict:
    attempts = []
    for tolerance in np.unique(np.linspace(args.ransac_residual, args.ransac_max_residual, 3)):
        fit = fit_boundary_line(points, (bounds[1] - bounds[0]) * args.min_x_coverage, float(tolerance),
                                x_bounds=bounds, max_rmse=args.ransac_residual,
                                min_inlier_ratio=args.min_inlier_ratio, max_gap_ratio=args.max_gap_ratio,
                                max_median_residual=args.max_median_residual)
        attempts.append({key: value for key, value in fit.items() if key not in {"points", "inliers"}})
        if fit["valid"]:
            break
    fit["attempts"] = attempts
    return fit


def reacquire_boundary(columns: list[dict], known: dict, side: str, points: list, h: int, args,
                       bounds: tuple | None = None) -> dict:
    bounds = bounds or (columns[0]["left"], columns[-1]["right"])
    offsets = [y - (known["slope"] * x + known["intercept"]) for x, y in points]
    offset = float(np.median(offsets))
    radius = max(2.0, h * args.recovery_band_ratio)
    candidates = []
    for column in columns:
        x = column["x"]
        anchor = known["slope"] * x + known["intercept"]
        predicted = anchor + offset
        evidence = boundary_evidence(column["profile"], column.get("brightness_edge"),
                                     args.texture_gap, args.texture_window, args.boundary_weights)[side]
        locations, _ = find_peaks(evidence, height=0.05, prominence=0.02,
                                  distance=max(2, min(args.smooth_kernel // 4, h // 80)))
        choices = []
        for y in locations:
            thickness = anchor - y if side == "top" else y - anchor
            if (abs(y - predicted) > radius
                    or not h * args.min_height_ratio <= thickness <= h * args.max_height_ratio):
                continue
            score = float(evidence[y] - 0.15 * abs(y - predicted) / radius)
            choices.append({"y0": float(y) if side == "top" else anchor,
                            "y1": anchor if side == "top" else float(y), "score": score})
        candidates.append({"x": x, "candidates": sorted(choices, key=lambda p: p["score"], reverse=True)[:12]})
    path = select_pair_path(candidates, h, args.continuity_weight, args.slope_weight)
    key = "y0" if side == "top" else "y1"
    fit = fit_boundary_with_retries([(p["x"], p[key]) for p in path], bounds, args)
    fit.update(reacquired=True, search_offset=offset, search_radius=radius)
    return fit


def reacquire_pair(columns: list[dict], top: dict, bottom: dict, shape: tuple, args) -> tuple:
    h, w = shape
    search = []
    for column in columns:
        choices = boundary_pair_candidates(column["profile"], args.min_height_ratio, args.max_height_ratio,
                                            args.smooth_kernel, column.get("brightness_edge"),
                                            args.texture_gap, args.texture_window, args.boundary_weights)
        for fit, key in ((top, "y0"), (bottom, "y1")):
            if "slope" in fit:
                predicted = fit["slope"] * column["x"] + fit["intercept"]
                choices = [p for p in choices if abs(p[key] - predicted) <= h * args.recovery_band_ratio]
        search.append({"x": column["x"], "candidates": choices})
    path = select_pair_path(search, h, args.continuity_weight, args.slope_weight)
    fits = []
    for key in ("y0", "y1"):
        fit = fit_boundary_with_retries([(p["x"], p[key]) for p in path], (0, w), args)
        fit["reacquired"] = True
        fits.append(fit)
    return tuple(fits)


def pair_quality(top: dict, bottom: dict, shape: tuple) -> dict:
    h, w = shape
    result = {}
    if all("slope" in fit for fit in (top, bottom)):
        heights = (bottom["slope"] - top["slope"]) * np.arange(w) + bottom["intercept"] - top["intercept"]
        result.update(height_mean_px=float(heights.mean()), height_std_px=float(heights.std()),
                      height_change_ratio=float(np.ptp(heights)) / h)
    supports = [{float(p[0]) for p, ok in zip(f.get("points", []), f.get("inliers", [])) if ok}
                for f in (top, bottom)]
    all_x = {float(p[0]) for f in (top, bottom) for p in f.get("points", [])}
    result["pair_inlier_ratio"] = len(supports[0] & supports[1]) / len(all_x) if all_x else 0.0
    return result


def fitted_pair_reason(top: dict, bottom: dict, shape: tuple, args) -> str | None:
    if not top["valid"] or not bottom["valid"]:
        return "insufficient_consistent_points"
    reason = boundary_geometry_reason(top, bottom, shape, args)
    quality = pair_quality(top, bottom, shape)
    if reason:
        return reason
    if quality["height_change_ratio"] > args.max_height_change_ratio:
        return "excessive_height_change"
    if quality["pair_inlier_ratio"] < args.min_pair_inlier_ratio:
        return "insufficient_joint_support"
    return None


def boundary_geometry_reason(top_line: dict, bottom_line: dict, shape: tuple, args) -> str | None:
    h, w = shape[:2]
    max_slope = np.tan(np.deg2rad(args.max_tilt_degrees))
    if max(abs(top_line["slope"]), abs(bottom_line["slope"])) > max_slope:
        return "tilt_exceeds_limit"
    endpoints = np.array([0, w - 1], dtype=np.float64)
    top = top_line["slope"] * endpoints + top_line["intercept"]
    bottom = bottom_line["slope"] * endpoints + bottom_line["intercept"]
    thickness = np.minimum(bottom, h) - np.maximum(top, 0)
    if np.any(thickness < max(1, h * args.min_height_ratio)):
        return "crossed_or_outside_boundaries"
    if np.any(bottom - top > h * args.max_height_ratio):
        return "height_exceeds_limit"
    return None


def rasterize_lines(result: dict, shape: tuple, args) -> None:
    h, w = shape[:2]
    xs = np.arange(w, dtype=np.float64)
    top_line, bottom_line = result["top_line"], result["bottom_line"]
    if getattr(args, 'mode', None) == 'gabor_boundary':
        mt, mb = args.boundary_config['mask']['margin_top'], args.boundary_config['mask']['margin_bottom']
        top_y = np.clip(np.rint(top_line['slope'] * xs + top_line['intercept']) - mt, 0, h).astype(np.int32)
        bottom_y = np.clip(np.rint(bottom_line['slope'] * xs + bottom_line['intercept']) + mb + 1, 0, h).astype(np.int32)
        result.update(top_y=top_y, bottom_y=bottom_y, y0=int(top_y.min()), y1=int(bottom_y.max()),
                      roi_margin_top=mt, roi_margin_bottom=mb, mask_rounding='nearest_inclusive')
        return
    top_y = np.clip(np.floor(top_line["slope"] * xs + top_line["intercept"] - args.margin),
                    0, h).astype(np.int32)
    bottom_y = np.clip(np.ceil(bottom_line["slope"] * xs + bottom_line["intercept"] + args.margin),
                       0, h).astype(np.int32)
    result.update(top_y=top_y, bottom_y=bottom_y, y0=int(top_y.min()), y1=int(bottom_y.max()))


def load_roi_overrides(path: Path | None, source: Path, reference_exposure: str | None) -> dict:
    if path is None:
        return {}
    try:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("root must be an object keyed by source-relative image paths")
        result = {}
        for name, modes in values.items():
            relative = Path(name)
            if relative.is_absolute() or relative.drive or ".." in relative.parts:
                raise ValueError(f"not a source-relative path: {name}")
            if not (source / relative).is_file():
                raise ValueError(f"override image does not exist: {name}")
            if reference_exposure and relative.parent.name.casefold() != reference_exposure.casefold():
                raise ValueError(f"override must name a {reference_exposure} reference image: {name}")
            if not isinstance(modes, dict) or not modes or set(modes) - set((*MODES, "auto", "gabor_boundary")):
                raise ValueError(f"invalid override modes: {name}")
            for mode, spec in modes.items():
                if isinstance(spec, dict) and ("top_points" in spec or "corners" in spec):
                    if __package__:
                        from .roi_annotations import line_annotation
                    else:
                        from roi_annotations import line_annotation
                    spec = modes[mode] = line_annotation(spec)
                if not isinstance(spec, dict) or set(spec) != {"shape", "top_y", "bottom_y"}:
                    raise ValueError(f"{name}/{mode}: expected shape, top_y and bottom_y")
                for field in ("shape", "top_y", "bottom_y"):
                    pair = spec[field]
                    if (not isinstance(pair, list) or len(pair) != 2
                            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                   or not np.isfinite(v) for v in pair)):
                        raise ValueError(f"{name}/{mode}: {field} must contain two finite numbers")
                if any(int(v) != v or v < 2 for v in spec["shape"]):
                    raise ValueError(f"{name}/{mode}: shape must contain integer height and width >= 2")
            key = relative.as_posix().casefold()
            if key in result:
                raise ValueError(f"duplicate override image: {name}")
            result[key] = modes
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"Invalid --roi-overrides: {exc}") from exc


def apply_roi_overrides(rois: dict, image_path: Path, shape: tuple, args, overrides: dict) -> None:
    relative = image_path.relative_to(args.source).as_posix().casefold()
    for mode, spec in overrides.get(relative, {}).items():
        if mode not in rois:
            continue
        h, w = shape[:2]
        if list(shape[:2]) != spec["shape"]:
            raise SystemExit(f"Manual ROI shape mismatch: {image_path} ({mode})")
        lines = {}
        for side in ("top", "bottom"):
            left, right = spec[f"{side}_y"]
            if not 0 <= min(left, right) <= max(left, right) <= h:
                raise SystemExit(f"Manual ROI boundary is outside image: {image_path} ({mode})")
            lines[side] = {"slope": float((right - left) / (w - 1)), "intercept": float(left)}
        reason = boundary_geometry_reason(lines["top"], lines["bottom"], shape, args)
        if reason:
            raise SystemExit(f"Invalid manual ROI: {image_path} ({mode}): {reason}")
        result = rois[mode]
        result.update(found=True, needs_review=False, fit_status="manual", fallback_reason=None,
                      no_roi_reason=None, manual_override=spec,
                      top_line=lines["top"], bottom_line=lines["bottom"])
        if mode in {"auto", "gabor_boundary"}:
            result.update(selection_method="manual", roi_detected=True, fallback_used=False,
                          output_warning=None, roi_quality=None)
        if mode == 'gabor_boundary':
            result.update(quality_status='MANUAL', needs_review_reason='', was_recovered=False,
                          recovery_method='manual_override', quality_details_json=None)
            for key in ('best_candidate_score', 'second_candidate_score', 'candidate_score_gap',
                        'candidate_boundary_gap_px', 'ambiguous_candidate_score', 'ambiguous_boundary_gap_px',
                        'outside_texture_top', 'outside_texture_bottom', 'consensus_count', 'consensus_total',
                        'consensus_ratio', 'quality_components_json'):
                result[key] = None
        for side in ("top", "bottom"):
            result[f"{side}_fit"] = {"valid": True, **lines[side], "method": "manual",
                                    "points": [[0, spec[f"{side}_y"][0]], [w - 1, spec[f"{side}_y"][1]]],
                                    "inliers": [True, True]}
        result.update(pair_quality(result["top_fit"], result["bottom_fit"], shape))
        result["pair_inlier_ratio"] = None
        rasterize_lines(result, shape, args)


def load_ground_truth(path: Path | None, source: Path, reference_exposure: str | None) -> dict:
    if path is None:
        return {}
    try:
        if path.suffix.casefold() == '.json':
            if __package__:
                from .roi_annotations import load_json_ground_truth
            else:
                from roi_annotations import load_json_ground_truth
            return load_json_ground_truth(path, source, reference_exposure)
        result, seen, splits = {}, set(), {}
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"source", "split", "status", "height", "width", "top_points_json", "bottom_points_json"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("GT CSV is missing required columns; see src/roi/ROI_USAGE.md")
            for row in reader:
                relative = Path(row["source"])
                if (not row["source"] or relative.is_absolute() or relative.drive or ".." in relative.parts
                        or not (source / relative).is_file()):
                    raise ValueError(f"invalid GT source: {row['source']}")
                if reference_exposure and relative.parent.name.casefold() != reference_exposure.casefold():
                    raise ValueError(f"GT must name a {reference_exposure} reference: {relative}")
                key = relative.as_posix().casefold()
                if key in seen:
                    raise ValueError(f"duplicate GT reference: {relative}")
                seen.add(key)
                if row["split"] not in {"tune", "validation"} or row["status"] not in {"pending", "approved"}:
                    raise ValueError(f"invalid GT split/status: {relative}")
                group = relative.parts[0].split("-", 1)[0].casefold()
                if group in splits and splits[group] != row["split"]:
                    raise ValueError(f"GT case appears in both tune and validation: {group}")
                splits[group] = row["split"]
                h, w = int(row["height"]), int(row["width"])
                if h < 2 or w < 2:
                    raise ValueError(f"invalid GT shape: {relative}")
                if row["status"] == "pending":
                    continue
                spec = {"shape": [h, w], "split": row["split"]}
                for side in ("top", "bottom"):
                    points = np.asarray(json.loads(row[f"{side}_points_json"]), dtype=float)
                    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                            or not np.isfinite(points).all()
                            or points[0, 0] != 0 or points[-1, 0] != w - 1
                            or np.any(np.diff(points[:, 0]) <= 0)
                            or np.any((points[:, 1] < 0) | (points[:, 1] > h))):
                        raise ValueError(f"invalid {side} GT points: {relative}")
                    spec[side] = points.tolist()
                xs = np.unique([p[0] for side in ("top", "bottom") for p in spec[side]])
                top, bottom = (np.asarray(spec[s]) for s in ("top", "bottom"))
                if np.any(np.interp(xs, top[:, 0], top[:, 1]) >= np.interp(xs, bottom[:, 0], bottom[:, 1])):
                    raise ValueError(f"crossed GT boundaries: {relative}")
                result[key] = spec
        print(f"ground_truth_approved={len(result)} pending={len(seen) - len(result)}", flush=True)
        return result
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f"Invalid --ground-truth: {exc}") from exc


def evaluate_ground_truth(rois: dict, image_path: Path, shape: tuple, args, annotations: dict) -> None:
    spec = annotations.get(image_path.relative_to(args.source).as_posix().casefold())
    if spec is None:
        return
    if list(shape[:2]) != spec["shape"]:
        raise SystemExit(f"GT shape mismatch: {image_path}")
    xs = np.arange(shape[1])
    for result in rois.values():
        evaluation = {"gt_split": spec["split"], "gt_auto_status": result["fit_status"],
                      "gt_auto_found": bool(result["found"] and not result.get("needs_review", False))}
        for side in ("top", "bottom"):
            # A rejected fit is not a usable prediction; do not silently score manual replacements.
            line = result.get(f"{side}_line")
            if line is not None and result[f"{side}_fit"]["valid"]:
                points = np.asarray(spec[side])
                gt = np.interp(xs, points[:, 0], points[:, 1])
                prediction = line["slope"] * xs + line["intercept"]
                evaluation[f"gt_{side}_mae_px"] = float(np.mean(np.abs(prediction - gt)))
        result["gt_evaluation"] = evaluation
        if getattr(args, 'mode', None) == 'gabor_boundary' and result.get('roi_detected'):
            if __package__:
                from .evaluate_roi_boundaries import boundary_metrics
            else:
                from evaluate_roi_boundaries import boundary_metrics
            evaluation.update({'gt_' + k: v for k, v in boundary_metrics(result, spec).items()})


def ground_truth_overrides(annotations: dict, modes: tuple) -> dict:
    overrides = {}
    for key, spec in annotations.items():
        if any(len(spec[s]) != 2 for s in ("top", "bottom")):
            raise SystemExit("--apply-ground-truth requires two endpoints per boundary; dense GT is evaluation-only.")
        manual = {"shape": spec["shape"], "top_y": [p[1] for p in spec["top"]],
                  "bottom_y": [p[1] for p in spec["bottom"]]}
        overrides[key] = {mode: manual for mode in modes}
    return overrides


def tilted_roi(
    coarse: dict, strip_rois: list[dict], shape: tuple[int, int], args,
    columns: list[dict] | None = None,
) -> dict:
    h, w = shape
    result = dict(coarse)
    result.update(margin=args.margin, fit_status="no_roi", fallback_reason=None, needs_review=True,
                  top_fit={"valid": False, "points": [], "inliers": []},
                  bottom_fit={"valid": False, "points": [], "inliers": []})
    if not coarse["found"]:
        result.update(top_y=np.zeros(w, dtype=np.int32), bottom_y=np.zeros(w, dtype=np.int32),
                      top_line=None, bottom_line=None)
        return result

    top_points, bottom_points = [], []
    for strip in strip_rois:
        if not strip["found"]:
            continue
        x = (strip["center_x0"] + strip["center_x1"] - 1) / 2
        top_points.append((x, float(strip["y0"])))
        bottom_points.append((x, float(strip["y1"])))
    bounds = (0, w)
    top_fit = fit_boundary_with_retries(top_points, bounds, args)
    bottom_fit = fit_boundary_with_retries(bottom_points, bounds, args)
    initial_fits = {"top": top_fit, "bottom": bottom_fit}
    if columns and top_fit["valid"] != bottom_fit["valid"]:
        if top_fit["valid"]:
            bottom_fit = reacquire_boundary(columns, top_fit, "bottom", bottom_points, h, args, bounds)
        else:
            top_fit = reacquire_boundary(columns, bottom_fit, "top", top_points, h, args, bounds)
    elif columns and fitted_pair_reason(top_fit, bottom_fit, shape, args):
        top_fit, bottom_fit = reacquire_pair(columns, top_fit, bottom_fit, shape, args)
    result["initial_fits"] = initial_fits
    result.update(top_fit=top_fit, bottom_fit=bottom_fit)
    result.update(pair_quality(top_fit, bottom_fit, shape))
    reason = fitted_pair_reason(top_fit, bottom_fit, shape, args)
    if reason is None:
        top_line = {key: top_fit[key] for key in ("slope", "intercept")}
        bottom_line = {key: bottom_fit[key] for key in ("slope", "intercept")}
        recovered = any(f.get("reacquired") or len(f["attempts"]) > 1 for f in (top_fit, bottom_fit))
        result.update(fit_status="recovered" if recovered else "fitted", needs_review=False)
    else:
        # Preserve the accepted side for diagnosis, but never export an invented band.
        top_line = {key: top_fit[key] for key in ("slope", "intercept")} if top_fit["valid"] else None
        bottom_line = {key: bottom_fit[key] for key in ("slope", "intercept")} if bottom_fit["valid"] else None
        result.update(found=False, fit_status="needs_review", fallback_reason=reason,
                      top_line=top_line, bottom_line=bottom_line,
                      top_y=np.zeros(w, dtype=np.int32), bottom_y=np.zeros(w, dtype=np.int32), y0=0, y1=0)
        return result
    result.update(top_line=top_line, bottom_line=bottom_line)
    rasterize_lines(result, shape, args)
    return result


def detect_image_rois(image: np.ndarray, modes: tuple[str, ...], args) -> dict:
    scores, x0, x1 = texture_row_scores(image, modes, args)
    coarse = {mode: detect_roi_y(score, 0, args.min_height_ratio, x0, x1,
                                max_height_ratio=args.max_height_ratio, smooth_kernel=args.smooth_kernel)
              for mode, score in scores.items()}
    columns = {mode: [] for mode in modes}
    minimum = 8 if x1 - x0 >= 64 else (4 if x1 - x0 >= 32 else 1)
    count = min(args.strips, max(minimum, (x1 - x0) // max(8, args.fft_window)))
    edges = np.linspace(x0, x1, count + 1, dtype=int)
    padding = max(args.std_kernel, args.gabor_kernel, args.fft_window) // 2
    gray = gray_float(image)
    scharr_y = np.abs(cv2.Scharr(gray, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_REFLECT))
    for start, end in zip(edges[:-1], edges[1:]):
        left, right = max(0, int(start) - padding), min(image.shape[1], int(end) + padding)
        local_scores, _, _ = texture_row_scores(
            image[:, left:right], modes, args, (int(start) - left, int(end) - left)
        )
        brightness = scharr_y[:, int(start):int(end)].mean(axis=1)
        smooth = odd_at_most(max(3, args.smooth_kernel // 3), image.shape[0])
        brightness = cv2.GaussianBlur(brightness.reshape(-1, 1), (1, smooth), 0).ravel()
        for mode in modes:
            candidates = boundary_pair_candidates(local_scores[mode], args.min_height_ratio,
                                                   args.max_height_ratio, args.smooth_kernel, brightness,
                                                   args.texture_gap, args.texture_window, args.boundary_weights)
            columns[mode].append({"x": float((start + end - 1) / 2), "candidates": candidates,
                                  "left": int(start), "right": int(end),
                                  "profile": local_scores[mode], "brightness_edge": brightness})
    results = {}
    for mode in modes:
        path = select_pair_path(columns[mode], image.shape[0], args.continuity_weight, args.slope_weight)
        if not coarse[mode]["found"] and len(path) >= 4:
            coarse[mode].update(found=True, y0=int(np.median([p["y0"] for p in path])),
                                y1=int(np.median([p["y1"] for p in path])))
        selected = [{"found": True, "center_x0": p["x"], "center_x1": p["x"] + 1,
                     "y0": p["y0"], "y1": p["y1"]} for p in path]
        result = tilted_roi(coarse[mode], selected, image.shape[:2], args, columns[mode])
        result.update(selected_pairs=path, candidate_counts=[len(c["candidates"]) for c in columns[mode]],
                      anchor_x=path[0].get("anchor_x") if path else None,
                      no_roi_reason=None if result["found"] else (
                          result["fallback_reason"] or "no_boundary_pair_within_height_limits"))
        results[mode] = result
    return results


def roi_pixel_mask(shape: tuple[int, int], roi: dict) -> np.ndarray:
    h, w = shape
    if not roi["found"]:
        return np.zeros((h, w), dtype=bool)
    top = roi.get("top_y", np.full(w, roi["y0"]))
    bottom = roi.get("bottom_y", np.full(w, roi["y1"]))
    ys = np.arange(h)[:, None]
    return (ys >= top) & (ys < bottom)


def mask_outside_roi(image: np.ndarray, roi: dict) -> np.ndarray:
    masked = np.zeros_like(image)
    if image.ndim == 3 and image.shape[2] == 4:
        # Black must be opaque in RGBA output, while ROI alpha remains unchanged.
        masked[:, :, 3] = np.iinfo(image.dtype).max
    if roi["found"]:
        keep = roi_pixel_mask(image.shape[:2], roi)
        masked[keep] = image[keep]
    return masked


def rectification_geometry(shape: tuple[int, int], roi: dict) -> dict | None:
    if not roi["found"] or shape[1] < 2:
        return None
    right = shape[1] - 1
    top, bottom = roi["top_line"], roi["bottom_line"]
    margin = roi["margin"]
    quad = np.array([
        [0, top["intercept"] - margin],
        [right, top["slope"] * right + top["intercept"] - margin],
        [right, bottom["slope"] * right + bottom["intercept"] + margin - 1],
        [0, bottom["intercept"] + margin - 1],
    ], dtype=np.float32)
    width = max(2, int(round(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))) + 1)
    height = max(2, int(round(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))) + 1)
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    transform = cv2.getPerspectiveTransform(quad, destination)
    return {"quad_xy": quad.tolist(), "output_size_wh": [width, height], "homography": transform.tolist()}


def display_bgr(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        scale = 255.0 / np.iinfo(image.dtype).max
        image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def draw_visualization(
    image: np.ndarray, roi: dict, mode: str, reference_note: str | None = None,
) -> np.ndarray:
    base = display_bgr(image)
    h, w = base.shape[:2]
    y0 = int(roi["y0"])
    y1 = int(roi["y1"])

    overlay = base.copy()
    mask = roi["mask"]
    overlay[roi_pixel_mask((h, w), roi)] = (255, 210, 80)
    base = cv2.addWeighted(overlay, 0.24, base, 0.76, 0)

    color = (0, 255, 80) if roi["found"] else (0, 160, 255)
    if roi["found"]:
        for boundary in (roi["top_y"], roi["bottom_y"] - 1):
            points = np.column_stack((np.arange(w), np.clip(boundary, 0, h - 1))).astype(np.int32)
            cv2.polylines(base, [points], False, color, 3, cv2.LINE_AA)
    else:
        for key in ("top_line", "bottom_line"):
            line = roi.get(key)
            if line is not None:
                xs = np.arange(w)
                ys = np.clip(line["slope"] * xs + line["intercept"], 0, h - 1)
                cv2.polylines(base, [np.column_stack((xs, ys)).astype(np.int32)], False, color, 3)
    for fit in (roi["top_fit"], roi["bottom_fit"]):
        for point, inlier in zip(fit["points"], fit["inliers"]):
            point_color = (0, 240, 255) if inlier else (0, 100, 255)
            cv2.circle(base, tuple(round(v) for v in point), 5, point_color, -1, cv2.LINE_AA)
    cv2.line(base, (roi["center_x0"], 0), (roi["center_x0"], h - 1), (180, 180, 180), 1)
    cv2.line(base, (roi["center_x1"], 0), (roi["center_x1"], h - 1), (180, 180, 180), 1)

    panel_w = max(240, min(420, w // 3))
    panel = np.full((h, panel_w, 3), 255, dtype=np.uint8)
    score_u8 = roi["score_u8"].ravel().astype(np.float32)
    x_score = 20 + (score_u8 / 255.0 * max(1, panel_w - 60)).astype(np.int32)
    panel[mask, :] = (245, 235, 210)
    cv2.line(panel, (0, y0), (panel_w - 1, y0), color, 2)
    cv2.line(panel, (0, max(y0, y1 - 1)), (panel_w - 1, max(y0, y1 - 1)), color, 2)

    points = np.column_stack([x_score, np.arange(h)]).astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(panel, [points], False, (25, 70, 200), 2, cv2.LINE_AA)

    status = roi["fit_status"].replace("_", " ")
    labels = [
        f"{MODE_LABELS[mode]} | Pair + DP + RANSAC | {status} | y={y0}:{y1} | margin={roi['margin']} px",
        "Yellow: inlier | Orange: rejected | Green: ROI with margin | Right: global row score",
    ]
    if reference_note:
        labels.append(reference_note)
    if not roi["found"]:
        labels.append(f"REVIEW REQUIRED: {roi.get('no_roi_reason')} | No mask exported")
    header = np.full((16 + 32 * len(labels), w + panel_w, 3), 245, dtype=np.uint8)
    for index, label in enumerate(labels):
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0]
        font_scale = 0.7 * min(1.0, (header.shape[1] - 32) / max(1, text_width))
        cv2.putText(header, label, (16, 30 + 32 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (25, 25, 25), 1, cv2.LINE_AA)
    return np.vstack([header, np.hstack([base, panel])])


def comparison_tile(masked: np.ndarray, roi: dict, mode: str) -> np.ndarray:
    base = display_bgr(masked)
    h, w = base.shape[:2]
    scale = min(640 / w, 640 / h)
    resized = cv2.resize(base, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    tile = np.full((688, 640, 3), 24, dtype=np.uint8)
    top = 48 + (640 - resized.shape[0]) // 2
    left = (640 - resized.shape[1]) // 2
    tile[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    status = (f"{roi['fit_status']} | y={roi['y0']}:{roi['y1']}"
              if roi["found"] else "needs review - no mask")
    label = f"{MODE_LABELS[mode]}  |  {status}"
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0]
    font_scale = 0.7 * min(1.0, (tile.shape[1] - 32) / max(1, text_width))
    cv2.putText(tile, label, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (240, 240, 240), 1, cv2.LINE_AA)
    return tile


def roi_csv_row(image: np.ndarray, image_path: Path, mode: str, result: dict, args,
                reference_path: Path | None = None) -> dict:
    row = result_identity(image_path, mode, args, reference_path or image_path)
    h, w = image.shape[:2]
    available = bool(result["found"])
    accepted = available and not result.get("needs_review", False)
    exportable = available and (accepted or result.get("allow_review_output", False))
    counts, pairs = result.get("candidate_counts", []), result.get("selected_pairs", [])
    row.update(status=result["fit_status"], reason=result.get("no_roi_reason"), found=available,
               needs_review=not accepted, input_height=h, input_width=w,
               auto_accepted=accepted and result['fit_status'] != 'manual', export_eligible=exportable,
               roi_detected=result.get("roi_detected", available),
               channels=1 if image.ndim == 2 else image.shape[2], dtype=str(image.dtype), margin=args.margin,
               selected_pair_count=len(pairs), candidate_column_count=len(counts),
               empty_candidate_columns=sum(n == 0 for n in counts),
               pair_score_mean=float(np.mean([p["score"] for p in pairs])) if pairs else None)
    if mode == 'gabor_boundary':
        for key in BOUNDARY_FIELDS:
            if key not in {'quality_details_json', 'automatic_prediction_json', 'manual_override'}:
                row[key] = result.get(key)
        row.update(scene_id=row['scene'], image_path=row['source'], exposure_us=int(row['exposure'][:-2]),
                   reference_image=row['roi_source'], manual_override=bool(result.get('manual_override')))
        mt, mb = result.get('roi_margin_top'), result.get('roi_margin_bottom')
        row['margin'] = mt if mt == mb else None
        if row['is_reference']:
            row.update(quality_details_json=result.get('quality_details_json'),
                       automatic_prediction_json=result.get('automatic_prediction_json'))
    for key in ("anchor_x", "height_mean_px", "height_std_px", "height_change_ratio", "pair_inlier_ratio"):
        row[key] = result.get(key)
    if row["is_reference"]:
        row.update(result.get("gt_evaluation", {}))
        for key in ("quality_components_json", "candidates_json", "fusion_sources_json", "registration_transform_json"):
            row[key] = result.get(key)
    for key in ("roi_quality", "roi_candidate_count", "selection_method", "fallback_used", "fusion_count", "consensus_count",
                "consensus_spread_px", "registration_source", "registration_correlation", "output_warning"):
        row[key] = result.get(key)
    for side in ("top", "bottom"):
        fit = result[f"{side}_fit"]
        points = np.asarray(fit.get("points", []), dtype=float).reshape(-1, 2)
        inliers = np.asarray(fit.get("inliers", []), dtype=bool)
        count = int(inliers.sum())
        span = float(np.ptp(points[inliers, 0])) if count else 0.0
        values = {
            "valid": fit["valid"], "point_count": len(points), "inlier_count": count,
            "inlier_ratio": count / len(points) if len(points) else None,
            "span_ratio": span / max(1, result["center_x1"] - result["center_x0"]),
            "rmse_px": fit.get("rmse"), "tolerance_px": fit.get("residual_threshold"),
            "attempt_count": len(fit.get("attempts", [])), "reacquired": fit.get("reacquired", False),
            "rejection_reason": fit.get("rejection_reason"),
            "initial_rejection_reason": result.get("initial_fits", {}).get(side, {}).get("rejection_reason"),
        }
        for key in ("x_coverage", "max_gap_ratio", "left_support", "center_support", "right_support",
                    "median_residual_px"):
            values[key] = fit.get(key)
        line = result.get(f"{side}_line")
        if line is not None:
            values.update(slope=line["slope"], angle_deg=float(np.rad2deg(np.arctan(line["slope"]))),
                          intercept=line["intercept"], y_left=line["intercept"],
                          y_right=line["slope"] * (w - 1) + line["intercept"])
        if row["is_reference"]:
            values.update(points_json=json.dumps(points.tolist()), inliers_json=json.dumps(inliers.tolist()))
        row.update({f"{side}_{key}": value for key, value in values.items()})
        if mode == 'gabor_boundary':
            row[side + '_ransac_rmse'] = fit.get('rmse')
            gap = fit.get('max_gap_ratio')
            row[side + '_longest_gap'] = gap * w if gap is not None else None
    if available:
        keep = roi_pixel_mask((h, w), result)
        # Target brightness statistics use only retained source pixels, never the black exterior.
        values = gray_float(image)[keep]
        channels = image[keep]
        if image.ndim == 3:
            channels = channels[:, :3]
        maximum = np.iinfo(image.dtype).max
        left = row["bottom_y_left"] - row["top_y_left"]
        right = row["bottom_y_right"] - row["top_y_right"]
        row.update(y0=result["y0"], y1_exclusive=result["y1"], roi_area_ratio=float(keep.mean()),
                   height_left=left, height_right=right, height_ratio_mean=(left + right) / (2 * h),
                   roi_gray_mean=float(values.mean()), roi_gray_std=float(values.std()),
                   roi_zero_fraction=float(np.mean(channels == 0)),
                   roi_saturated_fraction=float(np.mean(channels == maximum)))
    return row


def save_outputs(
    image: np.ndarray, image_path: Path, rois: dict, args,
    reference_path: Path | None = None, report: CsvReport | None = None,
) -> list[str]:
    relative = image_path.relative_to(args.source)
    summaries = []
    for mode, result in rois.items():
        row = roi_csv_row(image, image_path, mode, result, args, reference_path)
        if row["export_eligible"]:
            output_relative = Path("crops") / mode / relative.parent / f"{relative.name}_roi.png"
            if args.image_output == "rectified":
                geometry = rectification_geometry(image.shape[:2], result)
                pixels = cv2.warpPerspective(image, np.asarray(geometry["homography"]),
                                             tuple(geometry["output_size_wh"]),
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                row["homography_json"] = json.dumps(geometry["homography"])
            else:
                pixels = mask_outside_roi(image, result)
                row["output_origin_y"] = result["y0"] if args.image_output == "crop" else 0
                if args.image_output == "crop":
                    pixels = pixels[result["y0"]:result["y1"]]
            row.update(output_path=output_relative.as_posix(), output_height=pixels.shape[0],
                       output_width=pixels.shape[1], output_written=not args.dry_run)
            if not args.dry_run:
                imwrite(args.output / output_relative, pixels)
            summaries.append(f"{mode}={result['y0']}:{result['y1']} {result['fit_status']}")
        else:
            summaries.append(f"{mode}=needs_review ({row['reason']}); image skipped")
        if report is not None:
            report.write(row)
    return summaries


def run_reference_batch(images: list[Path], modes: tuple[str, ...], args, report: CsvReport,
                        overrides: dict | None = None, ground_truth: dict | None = None) -> None:
    jobs, unmatched = reference_jobs(images, args.reference_exposure)
    for item in unmatched:
        record_skipped(report, item["source"], modes, args, item["reason"])
    if not jobs:
        available = ", ".join(sorted({path.parent.name for path in images}))
        raise SystemExit(f"No reference images at {args.reference_exposure}. Available exposures: {available}")
    if args.limit is not None:
        jobs = jobs[:args.limit]
    stats = {
        "processed_reference_images": 0, "processed_images": 0,
        "skipped_images": 0, "no_roi_mode_outputs": 0, "needs_review_references": 0,
    }
    target_count = sum(len(targets) for _, targets in jobs)
    print(f"references={len(jobs)} targets={target_count} reference_exposure={args.reference_exposure} "
          f"modes={','.join(modes)}", flush=True)

    def skip(path: Path, reason: str, reference: Path, image=None):
        stats["skipped_images"] += 1
        record_skipped(report, path, modes, args, reason, reference, image)
        print(f"skip {reason}: {path}", flush=True)

    for index, (reference_path, targets) in enumerate(jobs, start=1):
        reference_image = imread(reference_path)
        if reference_image is None:
            for path in targets:
                skip(path, "unreadable_reference", reference_path)
            continue
        rois = detect_image_rois(reference_image, modes, args)
        evaluate_ground_truth(rois, reference_path, reference_image.shape, args, ground_truth or {})
        apply_roi_overrides(rois, reference_path, reference_image.shape, args, overrides or {})
        stats["needs_review_references"] += sum(not result["found"] for result in rois.values())
        stats["processed_reference_images"] += 1
        print(f"reference {index}/{len(jobs)}: {reference_path}", flush=True)
        for image_path in targets:
            image = reference_image if image_path == reference_path else imread(image_path)
            if image is None:
                skip(image_path, "unreadable_target", reference_path)
                continue
            if image.shape[:2] != reference_image.shape[:2]:
                skip(image_path, "reference_size_mismatch", reference_path, image)
                continue
            summaries = save_outputs(image, image_path, rois, args, reference_path, report)
            stats["processed_images"] += 1
            stats["no_roi_mode_outputs"] += sum(not result["found"] for result in rois.values())
            print(f"  {stats['processed_images']}/{target_count} {image_path.parent.name}/{image_path.name} "
                  + " | ".join(summaries), flush=True)
    print(f"processed_references={stats['processed_reference_images']}")
    print(f"processed={stats['processed_images']}")
    print(f"unmatched={len(unmatched)} skipped={stats['skipped_images']}")
    print(f"no_roi={stats['no_roi_mode_outputs']}")
    print(f"needs_review_references={stats['needs_review_references']}")
    print(f"output={args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit tilted texture boundaries and black out their exterior at original size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=Path("data/0918"))
    parser.add_argument("--output", type=Path, default=Path("data/roied"))
    parser.add_argument("--prefix", default="mono_")
    parser.add_argument("--extensions", default=",".join(sorted(IMAGE_EXTENSIONS)))
    exposure_options = parser.add_mutually_exclusive_group()
    exposure_options.add_argument("--exposure", default="10000us",
                                  help="Process this exposure independently")
    exposure_options.add_argument("--reference-exposure",
                                  help="Detect ROI at this exposure once, then apply to all exposures in the same scene/camera")
    parser.add_argument("--mode", type=str.lower, choices=(*MODES, "all", "auto", "gabor_boundary"), default="all",
                        help="Texture method; each mode writes ROI images and rows in one CSV")
    parser.add_argument("--std-kernel", type=int, default=31)
    parser.add_argument("--fft-window", type=int, default=64, help="Local square FFT window in pixels")
    parser.add_argument("--fft-stride", type=int, default=16)
    parser.add_argument("--fft-min-frequency", type=float, default=1 / 32,
                        help="Lower FFT band edge in cycles/pixel")
    parser.add_argument("--fft-max-frequency", type=float, default=1 / 4,
                        help="Upper FFT band edge in cycles/pixel")
    parser.add_argument("--gabor-kernel", type=int, default=51)
    parser.add_argument("--gabor-wavelengths", type=float, nargs="+", default=[8, 16, 32],
                        help="Gabor wavelengths in pixels; four orientations at each wavelength")
    parser.add_argument("--smooth-kernel", type=int, default=51)
    parser.add_argument("--texture-gap", type=int, default=15,
                        help="Exclude this many pixels around a boundary from texture comparison")
    parser.add_argument("--texture-window", type=int, default=60,
                        help="Outer distance of each inside/outside texture window in pixels")
    parser.add_argument("--boundary-weights", type=float, nargs=3, default=[0.45, 0.15, 0.40],
                        metavar=("TEXTURE_EDGE", "BRIGHTNESS", "CONTRAST"),
                        help="Relative candidate weights, normalized to sum to one")
    parser.add_argument("--center-x-ratio", type=float, default=0.90)
    parser.add_argument("--margin", type=int, default=20,
                        help="Vertical padding outside each fitted boundary in pixels")
    parser.add_argument("--min-height-ratio", type=float, default=0.32,
                        help="Minimum top-to-bottom boundary distance as a fraction of image height")
    parser.add_argument("--max-height-ratio", type=float, default=0.55,
                        help="Maximum top-to-bottom boundary distance as a fraction of image height")
    parser.add_argument("--strips", type=int, default=24,
                        help="Number of X intervals for local boundary detection")
    parser.add_argument("--continuity-weight", type=float, default=8.0,
                        help="DP penalty for changes in boundary positions and band height")
    parser.add_argument("--slope-weight", type=float, default=12.0,
                        help="Second-order DP penalty for abrupt boundary slope changes")
    parser.add_argument("--ransac-residual", type=float, default=8.0,
                        help="RANSAC inlier distance in pixels")
    parser.add_argument("--ransac-max-residual", type=float, default=16.0,
                        help="Upper limit for three RANSAC attempts; RMSE must still meet the base tolerance")
    parser.add_argument("--min-inlier-ratio", type=float, default=0.55)
    parser.add_argument("--min-x-coverage", type=float, default=0.75,
                        help="Minimum supported X span divided by full image width")
    parser.add_argument("--max-gap-ratio", type=float, default=0.25,
                        help="Maximum spacing between inliers, including gaps at image ends / full width")
    parser.add_argument("--max-median-residual", type=float, default=8.0)
    parser.add_argument("--max-height-change-ratio", type=float, default=0.15,
                        help="Maximum endpoint thickness change / image height, before margin")
    parser.add_argument("--min-pair-inlier-ratio", type=float, default=0.40,
                        help="Minimum fraction of sampled X positions supporting both fitted boundaries")
    parser.add_argument("--recovery-band-ratio", type=float, default=0.10,
                        help="Failed-side search radius as a fraction of image height")
    parser.add_argument("--roi-overrides", type=Path,
                        help="JSON manual reference boundaries; see src/roi/ROI_USAGE.md")
    parser.add_argument("--ground-truth", "--evaluate-gt", type=Path,
                        help="Approved manual GT CSV or JSON; evaluate automatic boundaries before overrides")
    parser.add_argument("--apply-ground-truth", action="store_true",
                        help="Also export approved two-endpoint GT as manual ROI for all selected modes")
    output_options = parser.add_mutually_exclusive_group()
    output_options.add_argument("--image-output", choices=("masked", "crop", "rectified"), default="masked",
                                help="masked preserves size; crop trims the Y bounding box; rectified resamples")
    output_options.add_argument("--rectify", dest="image_output", action="store_const", const="rectified",
                                help="Alias for --image-output rectified; exports only one image per result")
    parser.add_argument("--max-tilt-degrees", type=float, default=20,
                        help="Maximum accepted absolute boundary angle; otherwise require review")
    parser.add_argument("--limit", type=int,
                        help="Limit input images, or reference images with all their targets in reference mode")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-max-size", type=int, default=1024,
                        help="Maximum analysis dimension for auto mode; exports keep source resolution")
    parser.add_argument("--fusion-exposures", type=int, default=0,
                        help="Auto/boundary: number of assisting exposures; 0 uses all compatible exposures")
    parser.add_argument("--auto-min-quality", type=float, default=0.55,
                        help="Auto confidence threshold; lower-scoring outputs remain marked needs_review")
    parser.add_argument("--auto-fine-band", type=float, default=0.06,
                        help="Auto fine-boundary search radius / height around local FFT coarse boundary")
    parser.add_argument("--registration-min-correlation", type=float, default=0.65)
    parser.add_argument('--boundary-config', type=Path, help='YAML config for gabor_boundary only')
    parser.add_argument('--save-debug', action='store_true', help='Save reference diagnostic panels in gabor_boundary mode')
    parser.add_argument('--roi-margin-top', type=int, help='Override gabor_boundary YAML top margin')
    parser.add_argument('--roi-margin-bottom', type=int, help='Override gabor_boundary YAML bottom margin')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == 'gabor_boundary':
        if __package__:
            from .roi_boundary_quality import load_config
        else:
            from roi_boundary_quality import load_config
        try:
            args.boundary_config = load_config(args.boundary_config)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            parser.error(str(exc))
        if args.image_output != 'masked':
            parser.error('gabor_boundary requires --image-output masked; source coordinates must be preserved')
        for side in ('top', 'bottom'):
            margin = getattr(args, 'roi_margin_' + side)
            if margin is not None:
                if margin < 0:
                    parser.error('ROI margins must be nonnegative')
                args.boundary_config['mask']['margin_' + side] = margin
        args.min_height_ratio = args.boundary_config['pair']['min_height_ratio']
        args.max_height_ratio = args.boundary_config['pair']['max_height_ratio']
        args.max_tilt_degrees = args.boundary_config['quality']['max_tilt_deg']
    elif args.boundary_config or args.save_debug or args.roi_margin_top is not None or args.roi_margin_bottom is not None:
        parser.error('Boundary config, debug and separate margins require --mode gabor_boundary')
    for name in ("std_kernel", "smooth_kernel", "gabor_kernel", "fft_stride"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.fft_window < 4 or args.fft_stride > args.fft_window:
        parser.error("--fft-window must be >= 4 and --fft-stride must not exceed it")
    if not 0 < args.fft_min_frequency < args.fft_max_frequency <= 0.5:
        parser.error("FFT frequencies must satisfy 0 < min < max <= 0.5 cycles/pixel")
    if not all(np.isfinite(v) and v >= 2 for v in args.gabor_wavelengths):
        parser.error("--gabor-wavelengths must be finite and >= 2 pixels")
    if not 0 < args.center_x_ratio <= 1 or not 0 < args.min_height_ratio < args.max_height_ratio <= 1:
        parser.error("--center-x-ratio must be in (0, 1]; height ratios must satisfy 0 < min < max <= 1")
    for name in ("continuity_weight", "slope_weight"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not np.isfinite(args.ransac_residual) or args.ransac_residual <= 0:
        parser.error("--ransac-residual must be finite and positive")
    if not np.isfinite(args.ransac_max_residual) or args.ransac_max_residual < args.ransac_residual:
        parser.error("--ransac-max-residual must be finite and >= --ransac-residual")
    if not 0 < args.recovery_band_ratio <= 0.25:
        parser.error("--recovery-band-ratio must be in (0, 0.25]")
    for name in ("min_inlier_ratio", "min_x_coverage", "max_gap_ratio", "max_height_change_ratio",
                 "min_pair_inlier_ratio"):
        if not 0 < getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    if not np.isfinite(args.max_median_residual) or args.max_median_residual <= 0:
        parser.error("--max-median-residual must be finite and positive")
    if not 0 <= args.texture_gap < args.texture_window:
        parser.error("Texture windows require 0 <= --texture-gap < --texture-window")
    if (not all(np.isfinite(v) and v >= 0 for v in args.boundary_weights)
            or args.boundary_weights[0] + args.boundary_weights[2] <= 0):
        parser.error("--boundary-weights must be finite, nonnegative and include texture evidence")
    if args.margin < 0 or (args.limit is not None and args.limit < 1):
        parser.error("--margin must be nonnegative; --limit must be positive")
    if args.strips < 4 or not 0 < args.max_tilt_degrees < 90:
        parser.error("--strips must be >= 4; --max-tilt-degrees must be in (0, 90)")
    if not args.exposure or Path(args.exposure).name != args.exposure:
        parser.error("--exposure must be a single folder name, e.g. 10000us")
    if args.reference_exposure is not None and not re.fullmatch(r"\d+us", args.reference_exposure, re.IGNORECASE):
        parser.error("--reference-exposure must be an exposure folder name, e.g. 10000us")
    if args.apply_ground_truth and (args.ground_truth is None or args.roi_overrides is not None):
        parser.error("--apply-ground-truth requires --ground-truth and cannot be combined with --roi-overrides")
    if args.mode in {"auto", "gabor_boundary"} and args.reference_exposure is None:
        parser.error(f"--mode {args.mode} requires --reference-exposure (e.g. 10000us)")
    if args.auto_max_size < 128 or args.fusion_exposures < 0:
        parser.error("--auto-max-size must be >= 128; --fusion-exposures must be nonnegative")
    if not 0 < args.auto_min_quality <= 1 or not 0 < args.registration_min_correlation <= 1:
        parser.error("Quality and registration thresholds must be in (0, 1]")
    if not 0 < args.auto_fine_band <= .25:
        parser.error("--auto-fine-band must be in (0, 0.25]")

    source = args.source
    output = args.output
    if not source.is_dir():
        raise SystemExit(f"Source folder does not exist: {source}")
    overrides = load_roi_overrides(args.roi_overrides, source, args.reference_exposure)
    ground_truth = load_ground_truth(args.ground_truth, source, args.reference_exposure or args.exposure)

    extensions = parse_extensions(args.extensions)
    exposure = None if args.reference_exposure is not None else args.exposure
    images = sorted(iter_images(source, output, extensions, args.prefix or None, exposure))
    if not images:
        raise SystemExit("No matching images found.")

    modes = MODES if args.mode == "all" else (args.mode,)
    if args.apply_ground_truth:
        overrides = ground_truth_overrides(ground_truth, modes)
    with CsvReport(args) as report:
        if args.mode == 'gabor_boundary':
            if __package__:
                from . import roi_boundary
            else:
                import roi_boundary
            roi_boundary.run_batch(images, args, report, overrides, ground_truth)
        elif args.mode == "auto":
            if __package__:
                from . import auto_roi_group
            else:
                import auto_roi_group
            auto_roi_group.run_batch(images, args, report, overrides, ground_truth)
        elif args.reference_exposure is not None:
            run_reference_batch(images, modes, args, report, overrides, ground_truth)
        else:
            run_independent_batch(images, modes, args, report, overrides, ground_truth)


def run_independent_batch(images: list[Path], modes: tuple, args, report: CsvReport, overrides: dict,
                          ground_truth: dict | None = None) -> None:
    if args.limit is not None:
        images = images[:args.limit]
    processed = 0
    skipped = 0
    not_found = 0
    print(f"images={len(images)} exposure={args.exposure} modes={','.join(modes)}", flush=True)

    for image_path in images:
        image = imread(image_path)
        if image is None:
            print(f"skip unreadable: {image_path}")
            record_skipped(report, image_path, modes, args, "unreadable_target", image_path)
            skipped += 1
            continue

        rois = detect_image_rois(image, modes, args)
        evaluate_ground_truth(rois, image_path, image.shape, args, ground_truth or {})
        apply_roi_overrides(rois, image_path, image.shape, args, overrides)
        not_found += sum(not result["found"] for result in rois.values())
        summaries = save_outputs(image, image_path, rois, args, report=report)

        processed += 1
        print(
            f"{processed}/{len(images)} {image_path} "
            + " | ".join(summaries), flush=True,
        )

    print(f"processed={processed}")
    print(f"skipped={skipped}")
    print(f"no_roi={not_found}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
