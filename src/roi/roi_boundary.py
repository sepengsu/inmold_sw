"""Gabor ROI boundary-pair detection with explicit false-pass prevention gates."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

from . import auto_roi_local_std as core
from . import roi_boundary_quality as quality
from .auto_roi_group import safe_read


MODE = 'gabor_boundary'


def robust_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    low, high = np.percentile(values, [1, 99])
    return np.clip((values - low) / max(float(high - low), 1e-8), 0, 1).astype(np.float32)


def positive_unit(values: np.ndarray) -> np.ndarray:
    return np.clip(values / max(float(np.percentile(np.abs(values), 99)), 1e-8), 0, 1)


def interval_means(profile: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    n = len(profile)
    starts, ends = np.clip(starts, 0, n).astype(int), np.clip(ends, 0, n).astype(int)
    cumulative = np.r_[0., np.cumsum(profile, dtype=np.float64)]
    return (cumulative[ends] - cumulative[starts]) / np.maximum(1, ends - starts)


def features(image: np.ndarray, config: dict) -> dict:
    gray = robust_unit(core.gray_float(image))
    pre, gc, cc = (config[k] for k in ('preprocess', 'gabor', 'candidate'))
    smooth = gray
    if pre['gaussian_sigma'] > 0:
        smooth = cv2.GaussianBlur(gray, (0, 0), pre['gaussian_sigma'], borderType=cv2.BORDER_REFLECT)
    prepared = smooth.copy()
    if pre['illumination_normalization']:
        prepared -= cv2.GaussianBlur(prepared, (0, 0), pre['illumination_sigma'], borderType=cv2.BORDER_REFLECT)
    if pre['clahe']:
        prepared = cv2.createCLAHE(2., (8, 8)).apply((robust_unit(prepared) * 255).astype(np.uint8)).astype(np.float32) / 255
    energy = np.zeros(gray.shape, np.float32)
    for theta in np.arange(gc['orientations']) * np.pi / gc['orientations']:
        kernel = cv2.getGaborKernel((gc['kernel_size'], gc['kernel_size']), gc['sigma'], theta,
                                   gc['wavelength'], gc['gamma'], 0, ktype=cv2.CV_32F)
        kernel -= kernel.mean()
        kernel /= max(float(np.abs(kernel).sum()), 1e-8)
        response = np.abs(cv2.filter2D(prepared, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT))
        np.maximum(energy, response, out=energy)
    energy = robust_unit(cv2.boxFilter(energy, -1, (gc['envelope_size'],) * 2, borderType=cv2.BORDER_REFLECT))
    edge = robust_unit(np.abs(cv2.Scharr(smooth, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_REFLECT)))
    mean = cv2.boxFilter(smooth, -1, (31, 31), borderType=cv2.BORDER_REFLECT)
    variance = cv2.boxFilter(smooth * smooth, -1, (31, 31), borderType=cv2.BORDER_REFLECT) - mean * mean
    std = robust_unit(np.sqrt(np.maximum(variance, 0)))
    h, w = gray.shape
    reduce = np.mean if cc['aggregation'] == 'mean' else np.median
    bounds = [(x, min(w, x + cc['x_bin_width'])) for x in range(0, w, cc['x_bin_width'])]
    result = {'shape': (h, w), 'bounds': bounds,
              'x': np.array([(a + b - 1) / 2 for a, b in bounds]), 'energy_map': energy, 'edge_map': edge}
    for name, feature in (('gabor', energy), ('std', std), ('brightness', edge)):
        profiles = np.array([reduce(feature[:, a:b], axis=1) for a, b in bounds], np.float32)
        kernel = core.odd_at_most(cc['smooth_size'], h)
        result[name] = cv2.GaussianBlur(profiles, (kernel, 1), 0, borderType=cv2.BORDER_REFLECT)
    result['texture'] = (1 - cc['std_weight']) * result['gabor'] + cc['std_weight'] * result['std']
    y = np.arange(h)
    contrasts, insides, outsides = {}, {}, {}
    for side, sign in (('top', 1), ('bottom', -1)):
        before = np.array([interval_means(p, y - cc['gap'] - cc['window'], y - cc['gap']) for p in result['texture']])
        after = np.array([interval_means(p, y + cc['gap'], y + cc['gap'] + cc['window']) for p in result['texture']])
        insides[side], outsides[side] = (after, before) if sign == 1 else (before, after)
        contrasts[side] = sign * (after - before)
        transition = np.array([positive_unit(sign * np.gradient(p)) if h > 1 else np.zeros_like(p)
                               for p in result['gabor']])
        difference = np.array([positive_unit(p) for p in contrasts[side]])
        local = cv2.blur(difference, (1, 3), borderType=cv2.BORDER_REFLECT)
        brightness = np.array([positive_unit(p) for p in result['brightness']])
        weights = np.asarray(cc['weights']) / sum(cc['weights'])
        scores = sum(a * b for a, b in zip(weights, (transition, brightness, difference, local)))
        scores[contrasts[side] < cc['min_contrast']] = 0
        scores[:, :cc['gap'] + cc['window']] = 0
        scores[:, max(0, h - cc['gap'] - cc['window']):] = 0
        result[side + '_evidence'] = scores
    result.update(contrast=contrasts, inside=insides, outside=outsides)
    return result


def peak_locations(profile: np.ndarray, count: int, config: dict, allowed: np.ndarray | None = None) -> list[int]:
    if allowed is not None:
        # Excluded peaks must not consume the top-k budget or suppress nearby allowed peaks.
        peaks, _ = find_peaks(profile, height=config['min_score'])
        peaks = peaks[allowed[peaks]]
        restricted = np.full(profile.shape, -np.inf, dtype=float)
        restricted[peaks] = profile[peaks]
        profile = restricted
    locations, properties = find_peaks(profile, height=config['min_score'], distance=config['peak_distance'])
    order = np.argsort(properties['peak_heights'])[-count:]
    return sorted(int(locations[i]) for i in order)


def make_columns(feature: dict, config: dict) -> list:
    h, _ = feature['shape']
    cc, pc = config['candidate'], config['pair']
    weights = np.asarray(pc['weights']) / sum(pc['weights'])
    columns = []
    for i, x in enumerate(feature['x']):
        tops = peak_locations(feature['top_evidence'][i], cc['top_k'], cc)
        bottoms = peak_locations(feature['bottom_evidence'][i], cc['bottom_k'], cc)
        pairs = []
        for top in tops:
            for bottom in bottoms:
                thickness = (bottom - top) / h
                if not pc['min_height_ratio'] <= thickness <= pc['max_height_ratio']:
                    continue
                boundary = .5 * (feature['top_evidence'][i, top] + feature['bottom_evidence'][i, bottom])
                height = 1 - abs(thickness - pc['expected_height_ratio']) / max(pc['max_height_ratio'] - pc['min_height_ratio'], 1e-6)
                # Use near-boundary differences, never require uniformly strong fabric interior.
                contrast = np.clip(min(feature['contrast']['top'][i, top], feature['contrast']['bottom'][i, bottom]) * 2, 0, 1)
                smooth = 1 - .5 * (feature['outside']['top'][i, top] + feature['outside']['bottom'][i, bottom])
                score = float(np.dot(weights, [boundary, height, contrast, smooth]))
                pairs.append({'y0': float(top), 'y1': float(bottom), 'score': score})
        columns.append({'x': float(x), 'candidates': pairs, 'tops': tops, 'bottoms': bottoms})
    return columns


def path_for(columns: list, height: int, config: dict) -> list:
    valid = [c for c in columns if c['candidates']]
    if not valid:
        return []
    if len(valid) == 1:
        return [{'x': valid[0]['x'], **max(valid[0]['candidates'], key=lambda p: p['score'])}]
    spacing = float(np.median(np.diff([c['x'] for c in columns])))
    dc = config['dp']
    costs, links = core.pair_dp_states(valid, height, dc['lambda_position'], dc['lambda_slope'],
                                      spacing, dc['lambda_height'])
    return core.trace_pair_path(valid, links, np.unravel_index(np.argmin(costs), costs.shape))


def fit_line(points: list, width: int, config: dict) -> dict:
    q = config['quality']
    fits = []
    for threshold in config['ransac']['thresholds']:
        fit = core.fit_boundary_line(points, width * q['min_x_coverage'], threshold, (0, width),
                                     q['max_rmse_px'], q['min_inlier_ratio'], q['max_gap_ratio'],
                                     q['max_median_residual_px'], min_third_support=q['min_third_support'])
        fits.append(fit)
        if not quality.fit_reasons(fit, 'boundary', q):
            break
    valid = [f for f in fits if not quality.fit_reasons(f, 'boundary', q)]
    best = valid[0] if valid else min(fits, key=lambda f: (len(quality.fit_reasons(f, 'boundary', q)),
                                                         f.get('rmse', float('inf'))))
    best['attempts'] = [{k: v for k, v in f.items() if k not in {'points', 'inliers', 'attempts'}} for f in fits]
    return best


def recover_side(feature: dict, known: dict, side: str, expected_height: float, config: dict) -> dict:
    h, w = feature['shape']
    cc, pc = config['candidate'], config['pair']
    radius = config['recovery']['search_margin']
    columns = []
    for i, x in enumerate(feature['x']):
        fixed = known['slope'] * x + known['intercept']
        predicted = fixed + (-expected_height if side == 'top' else expected_height)
        ys = np.arange(h)
        heights = fixed - ys if side == 'top' else ys - fixed
        allowed = ((np.abs(ys - predicted) <= radius) & (heights >= pc['min_height_ratio'] * h)
                   & (heights <= pc['max_height_ratio'] * h))
        locations = peak_locations(feature[side + '_evidence'][i], max(cc['top_k'], cc['bottom_k']) * 3,
                                   cc, allowed)
        choices = []
        for y in locations:
            thickness = fixed - y if side == 'top' else y - fixed
            if abs(y - predicted) <= radius and pc['min_height_ratio'] * h <= thickness <= pc['max_height_ratio'] * h:
                choices.append({'y0': float(y) if side == 'top' else fixed,
                                'y1': fixed if side == 'top' else float(y),
                                'score': float(feature[side + '_evidence'][i, y])})
        columns.append({'x': float(x), 'candidates': choices})
    path = path_for(columns, h, config)
    key = 'y0' if side == 'top' else 'y1'
    fit = fit_line([(p['x'], p[key]) for p in path], w, config)
    fit.update(reacquired=True, search_radius=radius, search_offset=expected_height)
    return fit


def fit_pair(path: list, feature: dict, config: dict, method: str) -> dict | None:
    h, w = feature['shape']
    fits = {side: fit_line([(p['x'], p[key]) for p in path], w, config)
            for side, key in (('top', 'y0'), ('bottom', 'y1'))}
    if not path:
        # A missing pair path must not discard an independently supported side.
        for side in ('top', 'bottom'):
            single = []
            for i, x in enumerate(feature['x']):
                peaks = peak_locations(feature[side + '_evidence'][i], config['candidate'][side + '_k'], config['candidate'])
                choices = [{'y0': float(y), 'y1': float(y), 'score': float(feature[side + '_evidence'][i, y])}
                           for y in peaks]
                single.append({'x': float(x), 'candidates': choices})
            points = path_for(single, h, config)
            fits[side] = fit_line([(p['x'], p['y0']) for p in points], w, config)
    good = {s: not quality.fit_reasons(fits[s], s, config['quality']) for s in fits}
    recovery = []
    if sum(good.values()) == 1:
        failed = 'bottom' if good['top'] else 'top'
        known = 'top' if good['top'] else 'bottom'
        expected = float(np.median([p['y1'] - p['y0'] for p in path])) if path else h * config['pair']['expected_height_ratio']
        recovered = recover_side(feature, fits[known], failed, expected, config)
        if not quality.fit_reasons(recovered, failed, config['quality']):
            fits[failed] = recovered
            recovery.append('reacquire_' + failed)
    good = {s: not quality.fit_reasons(fits[s], s, config['quality']) for s in fits}
    for side, fit in fits.items():
        if fit.get('residual_threshold', 0) > config['ransac']['thresholds'][0]:
            recovery.append('relaxed_ransac_' + side)
    if not any('slope' in fit for fit in fits.values()):
        return None
    missing = [s for s in fits if 'slope' not in fits[s]]
    for side in missing:
        # Retain the supported side; an unknown side leaves the canvas untrimmed.
        fits[side].update(slope=0., intercept=0. if side == 'top' else float(h - 1), valid=False)
    candidate = {'selection_method': method, 'selected_pairs': path, 'roi_detected': not missing,
                 'was_recovered': bool(recovery), 'recovery_method': ';'.join(recovery)}
    for side, fit in fits.items():
        candidate[side + '_fit'] = fit
        candidate[side + '_line'] = {k: fit[k] for k in ('slope', 'intercept')}
    values = quality.line_values(candidate, w).reshape(2, 2)
    if not np.isfinite(values).all() or np.any(values[0] >= values[1]):
        if sum(good.values()) != 1:
            return None
        known = 'top' if good['top'] else 'bottom'
        failed = 'bottom' if good['top'] else 'top'
        known_values = values[0 if known == 'top' else 1]
        if not np.isfinite(known_values).all() or np.any(known_values <= 0) or np.any(known_values >= h - 1):
            return None
        rejected = fits[failed]
        line = {'slope': 0., 'intercept': 0. if failed == 'top' else float(h - 1)}
        fits[failed] = {'valid': False, **line, 'points': [], 'inliers': [],
                        'rejection_reason': 'invalid_pair_geometry', 'rejected_fit': rejected}
        candidate.update(roi_detected=False, **{failed + '_fit': fits[failed], failed + '_line': line})
    candidate.update(core.pair_quality(fits['top'], fits['bottom'], (h, w)))
    return candidate


def candidate_score(candidate: dict, feature: dict, config: dict, anchors: list) -> float:
    h, w = feature['shape']
    evidence = []
    for side in ('top', 'bottom'):
        line = candidate[side + '_line']
        ys = line['slope'] * feature['x'] + line['intercept']
        evidence.append(np.mean([np.interp(y, np.arange(h), p, left=0, right=0)
                                 for y, p in zip(ys, feature[side + '_evidence'])]))
    fits = [candidate[s + '_fit'] for s in ('top', 'bottom')]
    line_score = min(float(np.exp(-f.get('rmse', h) / config['quality']['max_rmse_px'])) for f in fits)
    coverage = min(f.get('x_coverage', 0) * (1 - min(1, f.get('max_gap_ratio', 1))) for f in fits)
    values = quality.line_values(candidate, w)
    agreement = np.mean([np.exp(-np.max(np.abs(values - a)) / config['quality']['consensus_distance_px']) for a in anchors]) if len(anchors) >= 2 else 0.
    geometry = float(np.exp(-candidate.get('height_change_ratio', 1) * h /
                            max(1, config['quality']['max_height_variation_px'])))
    parts = {'boundary': float(min(evidence)), 'line': line_score, 'coverage': float(coverage),
             'consensus': float(agreement), 'geometry': geometry}
    weights = np.asarray(config['quality']['weights']) / sum(config['quality']['weights'])
    score = float(np.dot(weights, list(parts.values())))
    candidate.update(roi_quality=score, quality_components=parts)
    return score


def detect_candidates(feature: dict, config: dict) -> tuple:
    columns = make_columns(feature, config)
    path = path_for(columns, feature['shape'][0], config)
    candidates = []
    for variant in ('primary', 'alternative_top', 'alternative_bottom', 'alternative_pair'):
        selected = path
        if variant != 'primary' and path:
            changed = copy.deepcopy(columns)
            anchors = {p['x']: p for p in path}
            keys = ('y0',) if variant.endswith('top') else ('y1',) if variant.endswith('bottom') else ('y0', 'y1')
            for col in changed:
                if col['x'] not in anchors:
                    continue
                for p in col['candidates']:
                    if all(abs(p[k] - anchors[col['x']][k]) < config['candidate']['alternative_radius'] for k in keys):
                        p['score'] -= config['candidate']['alternative_penalty']
            selected = path_for(changed, feature['shape'][0], config)
        candidate = fit_pair(selected, feature, config, 'gabor_' + variant)
        if candidate is not None:
            candidate['candidate_counts'] = [len(c['candidates']) for c in columns]
            candidate_score(candidate, feature, config, [])
            if not any(np.max(np.abs(quality.line_values(candidate, feature['shape'][1]) -
                                     quality.line_values(c, feature['shape'][1]))) <= 1 for c in candidates):
                candidates.append(candidate)
    return candidates, columns


def outside_check(candidate: dict, feature: dict, config: dict) -> dict:
    oc = config['outside']
    h, _ = feature['shape']
    result = {}
    for side, direction in (('top', -1), ('bottom', 1)):
        line = candidate[side + '_line']
        ratios, flags, valid = [], [], []
        for x, p in zip(feature['x'], feature['texture']):
            y = line['slope'] * x + line['intercept']
            intervals = []
            for sign in (direction, -direction):
                values = sorted([y + sign * oc['gap'], y + sign * (oc['gap'] + oc['window'])])
                intervals.append((int(round(values[0])), int(round(values[1]))))
            usable = all(a >= 0 and b <= h and b > a for a, b in intervals)
            valid.append(usable)
            if not usable:
                ratios.append(None)
                flags.append(False)
                continue
            out, inside = [float(p[a:b].mean()) for a, b in intervals]
            ratio = out / max(inside, oc['min_inside_energy'], 1e-6)
            ratios.append(ratio)
            # Weak local interior evidence cannot excuse strong texture outside.
            flags.append(out >= oc['min_energy'] and ratio >= oc['ratio_threshold'])
        longest, run = 0, 0
        for flag in flags:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        fraction = sum(flags) / max(1, sum(valid))
        available = [r for r in ratios if r is not None]
        result.update({side + '_ratio': float(np.percentile(available, 90)) if available else None,
                       side + '_flag_fraction': fraction, side + '_longest_run': longest,
                       side + '_ratios': ratios, side + '_flags': flags,
                       side + '_unavailable': bool(np.mean(valid) < oc['min_valid_fraction']),
                       side + '_clipped': bool(fraction >= oc['min_bin_fraction'] or longest >= oc['min_adjacent_bins'])})
    return result


def full_frame(shape: tuple) -> dict:
    result = {'selection_method': 'full_frame_preserved', 'roi_quality': 0., 'quality_components': {},
              'was_recovered': False, 'recovery_method': '', 'roi_detected': False}
    for side, b in (('top', 0.), ('bottom', float(shape[0] - 1))):
        line = {'slope': 0., 'intercept': b}
        result[side + '_line'] = line
        result[side + '_fit'] = {'valid': False, **line, 'points': [], 'inliers': []}
    return result


def consensus_anchor(candidates: list, feature: dict, config: dict) -> np.ndarray | None:
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c['roi_quality'])
    if not best.get('roi_detected'):
        return None
    validation = quality.assess(best, candidates, feature['shape'], config,
                                outside_check(best, feature, config), 0, 0)
    if validation['quality_status'] == 'NEEDS_REVIEW':
        return None
    return quality.line_values(best, feature['shape'][1])


def finalize(candidates: list, feature: dict, config: dict, anchors: list, total: int, args) -> dict:
    h, w = feature['shape']
    for candidate in candidates:
        candidate_score(candidate, feature, config, anchors)
    candidates.sort(key=lambda c: c['roi_quality'], reverse=True)
    candidate = candidates[0] if candidates else full_frame((h, w))
    values = quality.line_values(candidate, w)
    count = sum(np.max(np.abs(values - a)) <= config['quality']['consensus_distance_px'] for a in anchors)
    outside = outside_check(candidate, feature, config)
    validation = quality.assess(candidate, candidates, (h, w), config, outside, count, total)
    alternative = validation.pop('alternative')
    result = copy.deepcopy(candidate)
    result.update(validation)
    status = validation['quality_status']
    result.update(found=True, needs_review=status == 'NEEDS_REVIEW', allow_review_output=True,
                  fit_status=status, margin=args.margin, center_x0=0, center_x1=w,
                  fallback_used=not result['roi_detected'], consensus_count=int(count), fusion_count=total,
                  roi_candidate_count=len(candidates), no_roi_reason=validation['needs_review_reason'] or None,
                  output_warning=validation['needs_review_reason'] or None,
                  quality_details_json=json.dumps(validation),
                  quality_components_json=json.dumps(result['quality_components']),
                  candidates_json=json.dumps([{'method': c['selection_method'], 'quality': c['roi_quality'],
                                               'endpoints_native_px': quality.line_values(c, w).tolist(),
                                               'components': c['quality_components']} for c in candidates]),
                  alternative=alternative)
    core.rasterize_lines(result, (h, w), args)
    snapshot = {k: result.get(k) for k in ('top_line', 'bottom_line', 'quality_status', 'roi_detected',
                                          'roi_margin_top', 'roi_margin_bottom', 'mask_rounding')}
    snapshot.update(quality_details=validation, quality_components=result['quality_components'],
                    consensus_count=int(count))
    result['automatic_prediction_json'] = json.dumps(snapshot)
    return result


def save_debug(image: np.ndarray, feature: dict, columns: list, result: dict, path, config: dict,
               automatic: dict | None = None) -> None:
    original = core.display_bgr(image)
    automatic = automatic if automatic is not None else result
    panels = []
    def add(title, panel):
        width = 640
        panel = cv2.resize(panel, (width, max(1, round(panel.shape[0] * width / panel.shape[1]))))
        canvas = cv2.copyMakeBorder(panel, 34, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(canvas)
    def lines(panel, candidate):
        for side, color in (('top', (0, 255, 0)), ('bottom', (255, 255, 0))):
            line = candidate[side + '_line']
            w = panel.shape[1]
            cv2.line(panel, (0, round(line['intercept'])), (w - 1, round(line['slope'] * (w - 1) + line['intercept'])), color, 3)
        return panel
    add('Original', original.copy())
    add('Gabor envelope', cv2.applyColorMap((feature['energy_map'] * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    add('Brightness Scharr Y', cv2.cvtColor((feature['edge_map'] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    raw = original.copy()
    for col in columns:
        for y in col['tops'] + col['bottoms']:
            cv2.circle(raw, (round(col['x']), y), 5, (0, 0, 255), -1)
    add('Raw top / bottom candidates', raw)
    dp = original.copy()
    for key, color in (('y0', (0, 255, 0)), ('y1', (255, 255, 0))):
        points = np.array([[round(p['x']), round(p[key])] for p in result.get('selected_pairs', [])], np.int32)
        if len(points):
            cv2.polylines(dp, [points], False, color, 3)
    add('Selected pair DP path', dp)
    fitted = original.copy()
    for side in ('top', 'bottom'):
        fit = automatic[side + '_fit']
        for p, ok in zip(fit.get('points', []), fit.get('inliers', [])):
            cv2.circle(fitted, tuple(round(v) for v in p), 6, (255, 255, 255) if ok else (0, 165, 255), -1)
    add('Automatic RANSAC inliers white / outliers orange', lines(fitted, automatic))
    add('Final top green / bottom cyan', lines(original.copy(), result))
    overlay = original.copy()
    mask = core.roi_pixel_mask(image.shape[:2], result)
    overlay[mask] = (overlay[mask] * .6 + np.array([40, 90, 40]) * .4).astype(np.uint8)
    overlay[~mask] //= 3
    add('Final mask overlay', lines(overlay, result))
    add('Alternative candidate' if result.get('alternative') else 'No distinct alternative',
        lines(original.copy(), result['alternative']) if result.get('alternative') else original.copy())
    bands = original.copy()
    y = np.arange(image.shape[0])[:, None]
    x = np.arange(image.shape[1])
    for side, sign in (('top', -1), ('bottom', 1)):
        line = automatic[side + '_line']
        ys = line['slope'] * x + line['intercept']
        near = ys + sign * config['outside']['gap']
        far = ys + sign * (config['outside']['gap'] + config['outside']['window'])
        band = (y >= np.minimum(near, far)) & (y < np.maximum(near, far))
        bands[band] = (bands[band] * .5 + np.array([0, 0, 255]) * .5).astype(np.uint8)
    add('Automatic outside validation bands', lines(bands, automatic))
    grid = np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)])
    header = np.zeros((110, grid.shape[1], 3), np.uint8)
    q = result.get('roi_quality')
    angles = [float(np.rad2deg(np.arctan(result[s + '_line']['slope']))) for s in ('top', 'bottom')]
    texts = [f"{result['quality_status']} quality={q if q is not None else 'manual'} angles={angles[0]:.2f},{angles[1]:.2f}",
             f"coverage={result['top_fit'].get('x_coverage')} / {result['bottom_fit'].get('x_coverage')} height variation={result.get('height_variation_px')}",
             f"score gap={result.get('candidate_score_gap')} boundary gap={result.get('candidate_boundary_gap_px')}",
             result.get('needs_review_reason', '')]
    for i, text in enumerate(texts):
        cv2.putText(header, text[:160], (10, 22 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
    core.imwrite(path, np.vstack([header, grid]))


def run_batch(images: list, args, report, overrides: dict, ground_truth: dict) -> None:
    jobs, unmatched = core.reference_jobs(images, args.reference_exposure)
    stats = Counter()
    for item in unmatched:
        core.record_skipped(report, item['source'], (MODE,), args, item['reason'])
        stats['skipped_images'] += 1
    if not jobs:
        raise SystemExit(f'No reference images at {args.reference_exposure}')
    if args.limit:
        jobs = jobs[:args.limit]
    config = args.boundary_config
    for index, (reference, targets) in enumerate(jobs, 1):
        original = safe_read(reference)
        if original is None:
            for target in targets:
                core.record_skipped(report, target, (MODE,), args, 'unreadable_reference', reference)
                stats['skipped_images'] += 1
            continue
        feature = features(original, config)
        candidates, columns = detect_candidates(feature, config)
        anchors, used = [], []
        selected = sorted(targets, key=lambda p: (int(p.parent.name[:-2]), str(p)))
        if args.fusion_exposures and len(selected) > args.fusion_exposures:
            others = [p for p in selected if p != reference]
            indices = np.linspace(0, len(others) - 1, max(0, args.fusion_exposures - 1), dtype=int)
            selected = [reference, *(others[i] for i in indices)]
        for target in selected:
            pixels = original if target == reference else safe_read(target)
            if pixels is None or pixels.shape[:2] != original.shape[:2]:
                continue
            used.append(target.relative_to(args.source).as_posix())
            if target == reference:
                choices = candidates
                anchor_feature = feature
            else:
                auxiliary = features(pixels, config)
                choices, _ = detect_candidates(auxiliary, config)
                anchor_feature = auxiliary
            anchor = consensus_anchor(choices, anchor_feature, config)
            if anchor is not None:
                anchors.append(anchor)
            del anchor_feature
            if target != reference:
                del auxiliary
        result = finalize(candidates, feature, config, anchors, len(used), args)
        result['fusion_sources_json'] = json.dumps(used, ensure_ascii=False)
        rois = {MODE: result}
        core.evaluate_ground_truth(rois, reference, original.shape, args, ground_truth)
        automatic = copy.deepcopy(result) if args.save_debug else None
        core.apply_roi_overrides(rois, reference, original.shape, args, overrides)
        if result.get('manual_override'):
            values = quality.line_values(result, original.shape[1]).reshape(2, 2)
            heights = values[1] - values[0]
            result.update(roi_height_left=float(heights[0]), roi_height_center=float(heights.mean()),
                          roi_height_right=float(heights[1]), height_variation_px=float(np.ptp(heights)),
                          angle_diff_deg=float(abs(np.rad2deg(np.arctan(result['top_line']['slope'])) -
                                                   np.rad2deg(np.arctan(result['bottom_line']['slope'])))))
        if args.save_debug:
            relative = reference.relative_to(args.source)
            debug = Path('debug') / MODE / relative.parent / (relative.name + '_debug.png')
            result['debug_path'] = debug.as_posix()
            if not args.dry_run:
                save_debug(original, feature, columns, result, args.output / debug, config, automatic)
        stats[result['quality_status']] += 1
        print(f"scene {index}/{len(jobs)} {reference.relative_to(args.source)} {result['quality_status']} "
              f"quality={result.get('roi_quality')} reasons={result['needs_review_reason']}", flush=True)
        for target in targets:
            pixels = original if target == reference else safe_read(target)
            if pixels is None or pixels.shape[:2] != original.shape[:2]:
                reason = 'unreadable_target' if pixels is None else 'reference_size_mismatch'
                core.record_skipped(report, target, (MODE,), args, reason, reference, pixels)
                stats['skipped_images'] += 1
                continue
            core.save_outputs(pixels, target, rois, args, reference, report)
            stats['processed_images'] += 1
    print('boundary_summary=' + json.dumps(dict(stats)), flush=True)
