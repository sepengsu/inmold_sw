"""Group-level ROI candidates, continuous scoring, and fallback exports."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from . import auto_roi_local_std as core


def normalize_image(image: np.ndarray, size: int) -> np.ndarray:
    gray = core.gray_float(image)
    scale = min(1.0, size / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, (max(2, round(gray.shape[1] * scale)), max(2, round(gray.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
    low, high = np.percentile(gray, (1, 99.5))
    if high - low < 1e-6:
        return np.zeros(gray.shape, np.uint8)
    return np.clip((gray - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def analysis_args(args, original_shape: tuple, shape: tuple):
    scaled = SimpleNamespace(**vars(args))
    scale = min(shape[0] / original_shape[0], shape[1] / original_shape[1])
    for field in ('std_kernel', 'gabor_kernel', 'smooth_kernel'):
        setattr(scaled, field, max(3, round(getattr(args, field) * scale)) | 1)
    scaled.fft_window = max(8, round(args.fft_window * scale))
    scaled.fft_stride = max(1, round(args.fft_stride * scale))
    scaled.fft_min_frequency = min(.45, args.fft_min_frequency / scale)
    scaled.fft_max_frequency = min(.5, args.fft_max_frequency / scale)
    scaled.gabor_wavelengths = [max(2., x * scale) for x in args.gabor_wavelengths]
    scaled.texture_gap = max(1, round(args.texture_gap * scale))
    scaled.texture_window = max(scaled.texture_gap + 1, round(args.texture_window * scale))
    scaled.ransac_residual = max(1., args.ransac_residual * scale)
    scaled.ransac_max_residual = max(scaled.ransac_residual, args.ransac_max_residual * scale)
    scaled.max_median_residual = max(1., args.max_median_residual * scale)
    scaled.margin = 0
    return scaled


def frame_features(image: np.ndarray, args, modes=('std', 'fft')) -> dict:
    h, w = image.shape
    trim = (1 - args.center_x_ratio) / 2
    x0, x1 = int(w * trim), max(int(w * (1 - trim)), int(w * trim) + 1)
    count = min(args.strips, max(4, (x1 - x0) // 8), x1 - x0)
    edges = np.linspace(x0, x1, count + 1, dtype=int)
    result = {'x': (edges[:-1] + edges[1:] - 1) / 2, 'bounds': list(zip(edges[:-1], edges[1:])),
              'shape': (h, w), 'brightness': [], **{m: [] for m in modes}}
    padding = max(args.std_kernel, args.gabor_kernel, args.fft_window) // 2
    derivative = np.abs(cv2.Scharr(image.astype(np.float32), cv2.CV_32F, 0, 1,
                                  borderType=cv2.BORDER_REFLECT))
    for start, end in result['bounds']:
        left, right = max(0, start - padding), min(w, end + padding)
        scores, _, _ = core.texture_row_scores(image[:, left:right], modes, args, (start - left, end - left))
        for mode in modes:
            result[mode].append(core.normalize_score(scores[mode]))
        brightness = derivative[:, start:end].mean(axis=1)
        kernel = core.odd_at_most(max(3, args.smooth_kernel // 3), h)
        brightness = cv2.GaussianBlur(brightness[:, None], (1, kernel), 0).ravel()
        result['brightness'].append(core.normalize_score(brightness))
    for key in (*modes, 'brightness'):
        result[key] = np.asarray(result[key], np.float32)
    return result


def fuse_features(features: list[dict]) -> dict:
    fused = {key: features[0][key] for key in ('x', 'bounds', 'shape')}
    fused['exposure_count'] = len(features)
    for key in ('std', 'fft', 'brightness'):
        fused[key] = np.median(np.stack([f[key] for f in features]), axis=0).astype(np.float32)
    return fused


def columns_for(features: dict, mode: str, args, coarse: list | None = None) -> list:
    columns = []
    anchors = {p['x']: p for p in coarse or []}
    h = features['shape'][0]
    for i, (x, bounds) in enumerate(zip(features['x'], features['bounds'])):
        candidates = core.boundary_pair_candidates(features[mode][i], args.min_height_ratio,
                                                   args.max_height_ratio, args.smooth_kernel,
                                                   features['brightness'][i], args.texture_gap,
                                                   args.texture_window, args.boundary_weights)
        if coarse is not None:
            anchor = anchors.get(x)
            candidates = [] if anchor is None else [p for p in candidates if
                max(abs(p['y0'] - anchor['y0']), abs(p['y1'] - anchor['y1'])) <= h * args.auto_fine_band]
        columns.append({'x': float(x), 'left': int(bounds[0]), 'right': int(bounds[1]),
                        'profile': features[mode][i], 'brightness_edge': features['brightness'][i],
                        'candidates': candidates})
    return columns


def soft_fit(points: list, width: int, args) -> dict:
    attempts = []
    for tolerance in np.unique(np.linspace(args.ransac_residual, args.ransac_max_residual, 3)):
        fit = core.fit_boundary_line(points, 0, float(tolerance), (0, width), max_rmse=float('inf'),
                                     min_inlier_ratio=0, max_gap_ratio=1,
                                     max_median_residual=float('inf'))
        # In auto mode these are scoring features, not independent vetoes.
        fit['valid'] = all(k in fit and np.isfinite(fit[k]) for k in ('slope', 'intercept', 'rmse'))
        fit['valid'] = bool(fit['valid'] and fit.get('inlier_count', 0) >= 4)
        if fit['valid']:
            fit['rejection_reason'] = None
        attempts.append(fit)
    valid = [f for f in attempts if f['valid']]
    if not valid:
        return attempts[-1]
    best = max(valid, key=lambda f: np.exp(-f['rmse'] / max(args.ransac_residual, 1))
               * (.5 * f['inlier_ratio'] + .5 * f.get('x_coverage', 0)))
    best['attempts'] = [{k: v for k, v in f.items() if k not in {'points', 'inliers', 'attempts'}} for f in attempts]
    return best


def make_candidate(top: dict, bottom: dict, shape: tuple, args, method: str, **metadata) -> dict | None:
    if not top['valid'] or not bottom['valid']:
        return None
    if core.boundary_geometry_reason(top, bottom, shape, args):
        return None
    quality = core.pair_quality(top, bottom, shape)
    if quality['height_change_ratio'] > args.max_height_change_ratio:
        return None
    return {'top_fit': top, 'bottom_fit': bottom,
            'top_line': {k: top[k] for k in ('slope', 'intercept')},
            'bottom_line': {k: bottom[k] for k in ('slope', 'intercept')},
            'selection_method': method, **quality, **metadata}


def candidate_from_columns(columns: list, shape: tuple, args, method: str) -> tuple:
    path = core.select_pair_path(columns, shape[0], args.continuity_weight, args.slope_weight)
    fits = [soft_fit([(p['x'], p[key]) for p in path], shape[1], args) for key in ('y0', 'y1')]
    candidate = make_candidate(*fits, shape, args, method, selected_pairs=path,
                               candidate_counts=[len(c['candidates']) for c in columns],
                               anchor_x=path[0].get('anchor_x') if path else None)
    return candidate, path


def parallel_candidate(path: list, features: dict, args, method: str) -> dict | None:
    middle = soft_fit([(p['x'], (p['y0']+p['y1'])/2) for p in path], features['shape'][1], args)
    if not middle['valid']:
        return None
    thickness = float(np.median([p['y1']-p['y0'] for p, ok in zip(path,middle['inliers']) if ok]))
    left = middle['intercept']
    right = left + middle['slope']*(features['shape'][1]-1)
    values = np.array([left-thickness/2, right-thickness/2, left+thickness/2, right+thickness/2])
    # This is an additional hypothesis, not a constraint imposed on all ROIs.
    return candidate_from_endpoints(values, features, args, method + '_parallel')


def endpoints(candidate: dict, width: int) -> np.ndarray:
    return np.array([candidate[s + '_line']['intercept'] + candidate[s + '_line']['slope'] * x
                     for s in ('top', 'bottom') for x in (0, width - 1)])


def candidate_from_endpoints(values: np.ndarray, features: dict, args, method: str, **metadata) -> dict | None:
    h, w = features['shape']
    fits = []
    for side, pair in zip(('top', 'bottom'), np.asarray(values).reshape(2, 2)):
        slope, intercept = float((pair[1] - pair[0]) / (w - 1)), float(pair[0])
        points, inliers = [], []
        for i, x in enumerate(features['x']):
            evidence = core.boundary_evidence(features['std'][i], features['brightness'][i],
                                              args.texture_gap, args.texture_window, args.boundary_weights)[side]
            y = slope * x + intercept
            radius = max(2, round(args.ransac_max_residual))
            lo, hi = max(0, round(y) - radius), min(h, round(y) + radius + 1)
            if hi <= lo:
                continue
            peak = lo + int(np.argmax(evidence[lo:hi]))
            points.append((float(x), float(peak)))
            inliers.append(bool(evidence[peak] >= .08 and abs(peak - y) <= args.ransac_max_residual))
        xs = np.array([p[0] for p, ok in zip(points, inliers) if ok])
        errors = np.array([p[1] - (slope * p[0] + intercept) for p, ok in zip(points, inliers) if ok])
        fits.append({'valid': True, 'slope': slope, 'intercept': intercept, 'points': points, 'inliers': inliers,
                     'inlier_count': len(xs), 'inlier_ratio': len(xs) / max(1, len(points)),
                     'inlier_span': float(np.ptp(xs)) if len(xs) else 0,
                     'x_coverage': float(np.ptp(xs)) / w if len(xs) else 0,
                     'max_gap_ratio': float(np.diff(np.r_[0, np.sort(xs), w]).max()) / w,
                     'rmse': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                     'median_residual_px': float(np.median(np.abs(errors))) if len(errors) else None,
                     'method': method, 'rejection_reason': None,
                     **{f'{s}_support': int(np.sum((xs >= j*w/3) & (xs < (j+1)*w/3)))
                        for j, s in enumerate(('left', 'center', 'right'))}})
    return make_candidate(*fits, features['shape'], args, method, **metadata)


def score_candidate(candidate: dict, features: dict, args, anchors: list[np.ndarray]) -> float:
    h, w = features['shape']
    texture, edge = [], []
    for i, x in enumerate(features['x']):
        top = candidate['top_line']['slope'] * x + candidate['top_line']['intercept']
        bottom = candidate['bottom_line']['slope'] * x + candidate['bottom_line']['intercept']
        gap, window = args.texture_gap, args.texture_window
        for mode in ('std', 'fft'):
            profile = features[mode][i]
            def mean(a, b):
                a, b = max(0, int(a)), min(h, int(b))
                return float(profile[a:b].mean()) if b > a else 0.
            inside = mean(top + window, bottom - window)
            outside = .5 * (mean(top - window, top - gap) + mean(bottom + gap, bottom + window))
            texture.append(float(np.clip(2 * (inside - outside), 0, 1)))
        evidence = core.boundary_evidence(features['std'][i], features['brightness'][i],
                                          gap, window, args.boundary_weights)
        radius = max(1, round(args.ransac_residual))
        for side, y in (('top', top), ('bottom', bottom)):
            lo, hi = max(0, round(y) - radius), min(h, round(y) + radius + 1)
            edge.append(float(evidence[side][lo:hi].max()) if hi > lo else 0.)
    fits = [candidate[s + '_fit'] for s in ('top', 'bottom')]
    line = np.mean([np.exp(-(f.get('rmse') if f.get('rmse') is not None else h) /
                           max(1., args.ransac_residual)) * f.get('inlier_ratio', 0) for f in fits])
    coverage = np.mean([min(1., f.get('x_coverage', 0) / args.min_x_coverage)
                        * np.exp(-max(0., f.get('max_gap_ratio', 1) - args.max_gap_ratio) / .15) for f in fits])
    values = endpoints(candidate, w)
    distances = np.array([np.mean(np.abs(values - other)) for other in anchors])
    consistency = float(np.mean(np.exp(-distances / max(2, .015*h)))) if len(distances) >= 2 else 0.
    consistency *= min(1., len(anchors) / max(1, features.get('exposure_count', len(anchors))))
    geometry = float(np.exp(-.5*(candidate.get('height_change_ratio', 0)/.06)**2))
    components = {'texture': float(np.mean(texture)), 'edge': float(np.mean(edge)), 'line': float(line),
                  'coverage': float(coverage), 'consistency': consistency,
                  'pair': float(candidate.get('pair_inlier_ratio', 0)), 'geometry': geometry}
    weights = {'texture': .25, 'edge': .20, 'line': .15, 'coverage': .10,
               'consistency': .15, 'pair': .05, 'geometry': .10}
    score = sum(weights[k] * components[k] for k in weights)
    candidate.update(roi_quality=float(score), quality_components=components)
    return float(score)


def consensus_candidate(anchors: list[np.ndarray], features: dict, args) -> tuple:
    if not anchors:
        return None, [], None
    values = np.asarray(anchors)
    center = np.median(values, axis=0)
    distances = np.max(np.abs(values - center), axis=1)
    good = distances <= max(2., features['shape'][0] * .025)
    selected = list(values[good])
    required = max(2, features.get('exposure_count', len(anchors)) // 2 + 1)
    if len(selected) < required:
        return None, selected, None
    center = np.median(selected, axis=0)
    spread = float(np.median(np.mean(np.abs(np.asarray(selected) - center), axis=1)))
    return candidate_from_endpoints(center, features, args, 'exposure_consensus'), selected, spread


def hough_candidates(image: np.ndarray, features: dict, args, coarse: list) -> list:
    h, w = image.shape
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(smoothed, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/720, max(20, w//10),
                           minLineLength=max(20, w//5), maxLineGap=max(8, w//30))
    if lines is None:
        return []
    sides = {'top': [], 'bottom': []}
    if not coarse:
        return []
    expected = {side: np.median([p[key] for p in coarse]) for side, key in (('top','y0'), ('bottom','y1'))}
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        if abs(int(x1) - int(x0)) < w / 5:
            continue
        slope = float(y1 - y0) / float(x1 - x0)
        if abs(slope) > np.tan(np.deg2rad(args.max_tilt_degrees)):
            continue
        intercept = float(y0 - slope*x0)
        middle = intercept + slope*(w-1)/2
        for side in sides:
            if abs(middle - expected[side]) <= .08*h:
                sides[side].append((abs(int(x1)-int(x0)), [intercept, intercept + slope*(w-1)]))
    result = []
    for _, top in sorted(sides['top'], reverse=True)[:4]:
        for _, bottom in sorted(sides['bottom'], reverse=True)[:4]:
            candidate = candidate_from_endpoints(np.array([*top,*bottom]), features, args, 'edge_hough')
            if candidate is not None:
                result.append(candidate)
    return result


def pixel_resize_transform(source_shape: tuple, target_shape: tuple) -> np.ndarray:
    sy, sx = target_shape[0] / source_shape[0], target_shape[1] / source_shape[1]
    return np.array([[sx, 0, .5*sx-.5], [0, sy, .5*sy-.5], [0, 0, 1.]])


def register_template(template: np.ndarray, image: np.ndarray, values: np.ndarray, args) -> tuple | None:
    if template.shape != image.shape or min(image.shape) < 8:
        return None
    h, w = image.shape
    scale = min(1., 512 / max(h, w))
    size = (max(2, round(w*scale)), max(2, round(h*scale)))
    source = cv2.resize(template, size).astype(np.float32) / 255
    target = cv2.resize(image, size).astype(np.float32) / 255
    if source.std() < .01 or target.std() < .01:
        return None
    try:
        shift, response = cv2.phaseCorrelate(source, target)
        warp = np.eye(2, 3, dtype=np.float32)
        if response > .05 and max(abs(shift[0])/size[0], abs(shift[1])/size[1]) < .15:
            warp[:, 2] = shift
        correlation, warp = cv2.findTransformECC(source, target, warp, cv2.MOTION_AFFINE,
                                                  (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 80, 1e-5),
                                                  None, 5)
    except cv2.error:
        return None
    singular = np.linalg.svd(warp[:, :2], compute_uv=False)
    if (not np.isfinite(warp).all() or correlation < args.registration_min_correlation
            or np.linalg.det(warp[:, :2]) <= 0 or singular.min() < .85 or singular.max() > 1.15):
        return None
    resize = pixel_resize_transform((h,w), (size[1],size[0]))
    transform = np.linalg.inv(resize) @ np.vstack([warp, [0, 0, 1]]) @ resize
    if max(abs(transform[0, 2])/w, abs(transform[1, 2])/h) > .15:
        return None
    # ECC maps template coordinates to input coordinates. Transfer points directly;
    # WARP_INVERSE_MAP is only needed when resampling the input back to the template.
    transferred = []
    for yl, yr in np.asarray(values).reshape(2, 2):
        points = np.array([[0, yl, 1], [w - 1, yr, 1]]) @ transform.T
        if abs(points[1, 0] - points[0, 0]) < w / 2:
            return None
        slope = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])
        intercept = points[0, 1] - slope*points[0, 0]
        transferred.extend((intercept, intercept + slope*(w-1)))
    return np.asarray(transferred), float(correlation), transform


def build_group(normalized: list[np.ndarray], shape: tuple, args) -> dict:
    local_args = analysis_args(args, shape, normalized[0].shape)
    features = [frame_features(image, local_args) for image in normalized]
    fused = fuse_features(features)
    image = np.median(np.stack(normalized), axis=0).astype(np.uint8)
    anchors = []
    for feature in features:
        choices = []
        for mode in ('std', 'fft'):
            candidate, _ = candidate_from_columns(columns_for(feature, mode, local_args), feature['shape'],
                                                  local_args, 'exposure_' + mode)
            if candidate is not None:
                score_candidate(candidate, fused, local_args, [])
                choices.append(candidate)
        if choices:
            best = max(choices, key=lambda c: c['roi_quality'])
            if best['quality_components']['texture'] > .05 and best['quality_components']['edge'] > .05:
                anchors.append(endpoints(best, image.shape[1]))
    consensus, consistent, spread = consensus_candidate(anchors, fused, local_args)
    candidates = []
    fft_candidate, coarse = candidate_from_columns(columns_for(fused, 'fft', local_args), image.shape,
                                                    local_args, 'fft_coarse')
    for method, mode, constraint in (('std', 'std', None), ('fft_std', 'std', coarse)):
        candidate, path = candidate_from_columns(columns_for(fused, mode, local_args, constraint), image.shape,
                                                 local_args, method)
        parallel = parallel_candidate(path, fused, local_args, method)
        candidates.extend(c for c in (candidate, parallel) if c is not None)
    candidates.extend(c for c in (fft_candidate, consensus) if c is not None)
    parallel = parallel_candidate(coarse, fused, local_args, 'fft')
    if parallel is not None:
        candidates.append(parallel)
    enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(image)
    gabor = frame_features(enhanced, local_args, ('gabor',))
    fused['gabor'] = gabor['gabor']
    for method, constraint in (('fft_gabor', coarse), ('gabor_fallback', None)):
        candidate, path = candidate_from_columns(columns_for(fused, 'gabor', local_args, constraint), image.shape,
                                                 local_args, method)
        parallel = parallel_candidate(path, fused, local_args, method)
        candidates.extend(c for c in (candidate, parallel) if c is not None)
    candidates.extend(hough_candidates(image, fused, local_args, coarse))
    for candidate in candidates:
        score_candidate(candidate, fused, local_args, consistent)
    return {'candidates': candidates, 'features': fused, 'image': image, 'args': local_args,
            'anchors': consistent, 'consensus_spread': spread, 'shape': shape}


def scale_candidate(candidate: dict, shape: tuple, analysis_shape: tuple) -> dict:
    result = copy.deepcopy(candidate)
    sx, sy = analysis_shape[1] / shape[1], analysis_shape[0] / shape[0]
    for side in ('top', 'bottom'):
        fit = result[side + '_fit']
        slope, intercept = fit['slope'], fit['intercept']
        fit['slope'] = slope * sx / sy
        fit['intercept'] = (intercept + .5 + slope*(.5*sx - .5)) / sy - .5
        fit['points'] = [[(x + .5)/sx - .5, (y + .5)/sy - .5] for x, y in fit.get('points', [])]
        for key in ('rmse', 'median_residual_px', 'residual_threshold'):
            if fit.get(key) is not None:
                fit[key] /= sy
        if fit.get('inlier_span') is not None:
            fit['inlier_span'] /= sx
        result[side + '_line'] = {k: fit[k] for k in ('slope', 'intercept')}
    result['selected_pairs'] = [{**p, 'x': (p['x']+.5)/sx-.5,
                                 'y0': (p['y0']+.5)/sy-.5, 'y1': (p['y1']+.5)/sy-.5}
                                for p in result.get('selected_pairs', [])]
    if result.get('anchor_x') is not None:
        result['anchor_x'] = (result['anchor_x']+.5)/sx-.5
    result.update(core.pair_quality(result['top_fit'], result['bottom_fit'], shape))
    return result


def finalize_group(group: dict, args) -> dict:
    h, w = group['shape'][:2]
    candidates = sorted(group['candidates'], key=lambda c: c['roi_quality'], reverse=True)
    summaries = [{'method': c['selection_method'], 'quality': c['roi_quality'],
                  'components': c['quality_components'],
                  'endpoints_analysis_px': endpoints(c, group['image'].shape[1]).tolist(),
                  'registration_source': c.get('registration_source'),
                  'registration_correlation': c.get('registration_correlation')}
                 for c in candidates]
    result = None
    for candidate in candidates:
        native = scale_candidate(candidate, (h, w), group['image'].shape)
        if core.boundary_geometry_reason(native['top_line'], native['bottom_line'], (h, w), args) is None:
            result = native
            break
    if result is None:
        candidates = []
    if not candidates:
        # No scene evidence means no justified crop. Keep the full frame instead
        # of inventing a centered band that could erase the inspection target.
        result = {'selection_method': 'full_frame_fallback', 'roi_quality': 0., 'quality_components': {}}
        for side, intercept in (('top', 0.), ('bottom', float(h))):
            line = {'slope': 0., 'intercept': intercept}
            result[side + '_line'] = line
            result[side + '_fit'] = {'valid': False, **line, 'points': [], 'inliers': [],
                                    'rejection_reason': 'no_supported_boundary'}
        result.update(core.pair_quality(result['top_fit'], result['bottom_fit'], (h, w)))
    method = result['selection_method']
    accepted = bool(candidates and result['roi_quality'] >= args.auto_min_quality)
    fallback = method.startswith('gabor_fallback') or method in {'edge_hough', 'registration', 'full_frame_fallback'}
    warning = None if accepted else ('no_localized_roi_full_frame_preserved' if not candidates
                                    else 'low_quality_candidate_exported_for_review')
    result.update(found=True, needs_review=not accepted, allow_review_output=True,
                  roi_detected=bool(candidates), fallback_used=fallback,
                  fit_status=('auto_fallback' if fallback else 'auto') if accepted else
                             ('full_frame_fallback' if not candidates else 'low_confidence'),
                  no_roi_reason=warning, output_warning=warning, margin=args.margin,
                  center_x0=round(w*(1-args.center_x_ratio)/2), center_x1=round(w*(1+args.center_x_ratio)/2),
                  quality_components_json=json.dumps(result['quality_components']),
                  candidates_json=json.dumps(summaries), roi_candidate_count=len(summaries),
                  fusion_count=len(group['sources']),
                  fusion_sources_json=json.dumps(group['sources'], ensure_ascii=False),
                  consensus_count=len(group['anchors']),
                  consensus_spread_px=(group['consensus_spread'] * h / group['image'].shape[0]
                                       if group['consensus_spread'] is not None else None))
    core.rasterize_lines(result, (h, w), args)
    return result


def safe_read(path: Path):
    try:
        return core.imread(path)
    except (OSError, cv2.error):
        return None


def compatible_template(first: dict, second: dict) -> bool:
    a, b = first['reference'], second['reference']
    return (first['shape'] == second['shape'] and first['image'].shape == second['image'].shape
            and a.stem.split('_')[0].casefold() == b.stem.split('_')[0].casefold()
            and a.parent.parent.name.casefold() == b.parent.parent.name.casefold()
            and first['case'] == second['case'])


def run_batch(images: list[Path], args, report, overrides: dict, ground_truth: dict) -> None:
    jobs, unmatched = core.reference_jobs(images, args.reference_exposure)
    for item in unmatched:
        core.record_skipped(report, item['source'], ('auto',), args, item['reason'])
    if not jobs:
        raise SystemExit(f'No reference images at {args.reference_exposure}')
    if args.limit is not None:
        jobs = jobs[:args.limit]
    groups = []
    print(f'auto references={len(jobs)} targets={sum(len(t) for _, t in jobs)}', flush=True)
    for index, (reference, targets) in enumerate(jobs, 1):
        image = safe_read(reference)
        if image is None:
            for target in targets:
                core.record_skipped(report, target, ('auto',), args, 'unreadable_reference', reference)
            continue
        shape = image.shape[:2]
        selected = sorted(targets, key=lambda p: (int(p.parent.name[:-2]), str(p)))
        if args.fusion_exposures and len(selected) > args.fusion_exposures:
            others = [p for p in selected if p != reference]
            indices = np.linspace(0, len(others)-1, max(0, args.fusion_exposures-1), dtype=int)
            selected = [reference, *(others[i] for i in indices)]
        normalized, used = [], []
        for path in selected:
            pixels = image if path == reference else safe_read(path)
            if pixels is not None and pixels.shape[:2] == shape:
                normalized.append(normalize_image(pixels, args.auto_max_size))
                used.append(path.relative_to(args.source).as_posix())
        group = build_group(normalized, shape, args)
        relative = reference.relative_to(args.source)
        group.update(reference=reference, targets=targets, sources=used,
                     case=relative.parts[0].split('-', 1)[0].casefold())
        groups.append(group)
        best = max(group['candidates'], key=lambda c: c['roi_quality']) if group['candidates'] else None
        print(f'detect {index}/{len(jobs)} {relative.as_posix()} candidates={len(group["candidates"])} '
              f'best={best["selection_method"] if best else "none"} '
              f'quality={best["roi_quality"] if best else 0:.3f}', flush=True)
    # Snapshot templates before any transfers so an uncertain transfer cannot propagate.
    templates = []
    for group in groups:
        if group['candidates']:
            best = max(group['candidates'], key=lambda c: c['roi_quality'])
            if best['roi_quality'] >= args.auto_min_quality:
                templates.append((group, best))
    stats = Counter()
    for group in groups:
        best_score = max((c['roi_quality'] for c in group['candidates']), default=0.)
        if best_score < args.auto_min_quality:
            choices = [(g, c) for g, c in templates if g is not group and compatible_template(group, g)]
            choices.sort(key=lambda pair: pair[1]['roi_quality'], reverse=True)
            for template, candidate in choices[:3]:
                transfer = register_template(template['image'], group['image'],
                                             endpoints(candidate, template['image'].shape[1]), args)
                if transfer is None:
                    continue
                values, correlation, transform = transfer
                registered = candidate_from_endpoints(values, group['features'], group['args'], 'registration',
                                                       registration_source=template['reference'].relative_to(args.source).as_posix(),
                                                       registration_correlation=correlation)
                if registered is not None:
                    scale = pixel_resize_transform(group['shape'], group['image'].shape)
                    registered['registration_transform_json'] = json.dumps((np.linalg.inv(scale) @ transform @ scale).tolist())
                    score_candidate(registered, group['features'], group['args'], group['anchors'])
                    group['candidates'].append(registered)
        result = finalize_group(group, args)
        rois = {'auto': result}
        core.evaluate_ground_truth(rois, group['reference'], group['shape'], args, ground_truth)
        core.apply_roi_overrides(rois, group['reference'], group['shape'], args, overrides)
        stats[result['fit_status']] += 1
        print(f'export {group["reference"]} {result["fit_status"]} method={result["selection_method"]} '
              f'quality={result.get("roi_quality")}', flush=True)
        for path in group['targets']:
            pixels = safe_read(path)
            if pixels is None:
                core.record_skipped(report, path, ('auto',), args, 'unreadable_target', group['reference'])
                stats['skipped_images'] += 1
            elif pixels.shape[:2] != group['shape']:
                core.record_skipped(report, path, ('auto',), args, 'reference_size_mismatch', group['reference'], pixels)
                stats['skipped_images'] += 1
            else:
                core.save_outputs(pixels, path, rois, args, group['reference'], report)
                stats['processed_images'] += 1
    print(f'auto_summary={json.dumps(dict(stats))}', flush=True)
    print(f'output={args.output}', flush=True)
