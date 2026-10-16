"""Native-pixel review gates; no single scalar score can override these checks."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'config' / 'roi_gabor_boundary.yaml'


def load_config(path: Path | None = None) -> dict:
    with DEFAULT_CONFIG.open(encoding='utf-8') as handle:
        defaults = yaml.safe_load(handle)
    config = copy.deepcopy(defaults)
    if path is not None:
        with path.open(encoding='utf-8-sig') as handle:
            override = yaml.safe_load(handle)
        if not isinstance(override, dict):
            raise ValueError('boundary config must be a mapping')
        for section, values in override.items():
            if section not in config or not isinstance(values, dict):
                raise ValueError(f'unknown or invalid config section: {section}')
            if set(values) - set(config[section]):
                raise ValueError(f'unknown config keys in {section}: {set(values) - set(config[section])}')
            config[section].update(values)
    for section, values in config.items():
        for key, value in values.items():
            expected = defaults[section][key]
            label = f'{section}.{key}'
            if isinstance(expected, bool):
                if not isinstance(value, bool):
                    raise ValueError(f'{label} must be boolean')
            elif isinstance(expected, str):
                if value not in {'mean', 'median'}:
                    raise ValueError(f'{label} must be mean or median')
            else:
                if isinstance(expected, list) != isinstance(value, list):
                    raise ValueError(f'{label} must be a list' if isinstance(expected, list)
                                     else f'{label} must be a scalar number')
                numbers = value if isinstance(value, list) else [value]
                if (isinstance(expected, list) and (not isinstance(value, list) or len(value) != len(expected))
                        or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                   and np.isfinite(v) and v >= 0 for v in numbers)):
                    raise ValueError(f'{label} must contain finite nonnegative numbers')
                if isinstance(expected, int) and (not isinstance(value, int) or isinstance(value, bool)):
                    raise ValueError(f'{label} must be an integer')
                if key == 'weights' and sum(numbers) <= 0:
                    raise ValueError(f'{label} cannot be all zero')
    for section, key in [('gabor', 'kernel_size'), ('gabor', 'envelope_size'), ('candidate', 'smooth_size')]:
        value = config[section][key]
        if value < 3 or value % 2 != 1:
            raise ValueError(f'{section}.{key} must be odd and >= 3')
    for section, keys in {'gabor': ['sigma', 'wavelength', 'gamma', 'orientations'],
                          'candidate': ['x_bin_width', 'top_k', 'bottom_k', 'window', 'peak_distance'],
                          'recovery': ['search_margin'], 'outside': ['window', 'min_adjacent_bins'],
                          'quality': ['max_rmse_px', 'max_tilt_deg', 'consensus_distance_px']}.items():
        if any(config[section][key] <= 0 for key in keys):
            raise ValueError(f'{section}: positive values required for {keys}')
    if config['preprocess']['illumination_sigma'] <= 0:
        raise ValueError('illumination_sigma must be positive')
    pair = config['pair']
    if not 0 < pair['min_height_ratio'] <= pair['expected_height_ratio'] <= pair['max_height_ratio'] <= 1:
        raise ValueError('pair heights must satisfy 0 < min <= expected <= max <= 1')
    thresholds = config['ransac']['thresholds']
    if thresholds[0] <= 0 or any(b <= a for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError('RANSAC thresholds must be positive and strictly increasing')
    for section, keys in {'quality': ['min_score', 'min_inlier_ratio', 'min_x_coverage', 'max_gap_ratio',
                                     'min_consensus_ratio'],
                          'outside': ['min_energy', 'min_inside_energy', 'min_bin_fraction', 'min_valid_fraction'],
                          'candidate': ['std_weight', 'min_score', 'min_contrast']}.items():
        if any(not 0 <= config[section][key] <= 1 for key in keys):
            raise ValueError(f'{section}: fraction values must be in [0, 1]')
    if config['quality']['max_tilt_deg'] >= 90:
        raise ValueError('max_tilt_deg must be less than 90')
    return config


def line_values(result: dict, width: int) -> np.ndarray:
    return np.array([result[side + '_line']['intercept'] + result[side + '_line']['slope'] * x
                     for side in ('top', 'bottom') for x in (0, width - 1)], dtype=float)


def ambiguity(candidate: dict, candidates: list, width: int, config: dict) -> dict:
    values = line_values(candidate, width)
    others = sorted((c for c in candidates if c is not candidate), key=lambda c: c['roi_quality'], reverse=True)
    # Duplicate geometry from different methods is not independent evidence.
    distinct = [(c, float(np.max(np.abs(line_values(c, width) - values)))) for c in others]
    distinct = [(c, gap) for c, gap in distinct if gap > 1.0]
    second = distinct[0] if distinct else None
    competing = next(((c, gap) for c, gap in distinct
                      if abs(candidate['roi_quality'] - c['roi_quality']) < config['ambiguity_score_gap']
                      and gap > config['ambiguity_boundary_gap_px']), None)
    selected = competing or second
    return {'best_candidate_score': candidate['roi_quality'],
            'second_candidate_score': second[0]['roi_quality'] if second else None,
            'candidate_score_gap': candidate['roi_quality'] - second[0]['roi_quality'] if second else None,
            'candidate_boundary_gap_px': second[1] if second else None,
            'ambiguous_candidate_score': competing[0]['roi_quality'] if competing else None,
            'ambiguous_boundary_gap_px': competing[1] if competing else None,
            'ambiguous': competing is not None, 'alternative': selected[0] if selected else None}


def fit_reasons(fit: dict, side: str, config: dict) -> list[str]:
    side = side.upper()
    reasons = []
    if not fit.get('valid', False):
        reasons.append('RANSAC_FAILED_' + side)
    for key, threshold, label, lower in (
        ('inlier_ratio', config['min_inlier_ratio'], 'LOW_INLIER_RATIO_', True),
        ('x_coverage', config['min_x_coverage'], 'LOW_X_COVERAGE_', True),
        ('max_gap_ratio', config['max_gap_ratio'], 'LONG_UNSUPPORTED_GAP_', False),
        ('rmse', config['max_rmse_px'], 'HIGH_RANSAC_RMSE_', False),
        ('median_residual_px', config['max_median_residual_px'], 'HIGH_MEDIAN_RESIDUAL_', False),
    ):
        value = fit.get(key)
        if value is None or not np.isfinite(value) or (value < threshold if lower else value > threshold):
            reasons.append(label + side)
    if any(fit.get(third + '_support', 0) < config['min_third_support'] for third in ('left', 'center', 'right')):
        reasons.append('MISSING_THIRD_SUPPORT_' + side)
    return reasons


def assess(candidate: dict, candidates: list, shape: tuple, config: dict, outside: dict,
           consensus_count: int, consensus_total: int) -> dict:
    h, w = shape[:2]
    q = config['quality']
    top, bottom = (candidate[s + '_line'] for s in ('top', 'bottom'))
    xs = np.array([0., (w - 1) / 2, w - 1])
    tv, bv = (line['slope'] * xs + line['intercept'] for line in (top, bottom))
    heights = bv - tv
    angles = np.rad2deg(np.arctan([top['slope'], bottom['slope']]))
    reasons = []
    if (not np.isfinite(np.r_[tv, bv]).all() or np.any(heights <= 0)
            or np.any(tv < 0) or np.any(bv >= h)):
        reasons.append('INVALID_GEOMETRY')
    if np.any(heights < h * config['pair']['min_height_ratio']) or np.any(heights > h * config['pair']['max_height_ratio']):
        reasons.append('IMPLAUSIBLE_HEIGHT')
    angle_diff = float(abs(angles[0] - angles[1]))
    variation = float(np.ptp(heights))
    if np.max(np.abs(angles)) > q['max_tilt_deg']:
        reasons.append('EXCESSIVE_TILT')
    if angle_diff > q['max_angle_diff_deg']:
        reasons.append('LARGE_ANGLE_DIFFERENCE')
    if variation > q['max_height_variation_px']:
        reasons.append('LARGE_HEIGHT_VARIATION')
    for side in ('top', 'bottom'):
        reasons.extend(fit_reasons(candidate[side + '_fit'], side, q))
        if outside.get(side + '_unavailable', True):
            reasons.append('OUTSIDE_CHECK_UNAVAILABLE_' + side.upper())
        if outside.get(side + '_clipped', False):
            reasons.append('FABRIC_OUTSIDE_' + side.upper())
    ambiguity_info = ambiguity(candidate, candidates, w, q)
    if ambiguity_info['ambiguous']:
        reasons.append('AMBIGUOUS_CANDIDATES')
    ratio = consensus_count / consensus_total if consensus_total else 0.
    if consensus_total > 1 and ratio < q['min_consensus_ratio']:
        reasons.append('LOW_MULTI_EXPOSURE_CONSENSUS')
    if not np.isfinite(candidate['roi_quality']) or candidate['roi_quality'] < q['min_score']:
        reasons.append('LOW_BOUNDARY_SCORE')
    recovered = bool(candidate.get('was_recovered'))
    return {**ambiguity_info, 'angle_diff_deg': angle_diff,
            'roi_height_left': float(heights[0]), 'roi_height_center': float(heights[1]),
            'roi_height_right': float(heights[2]), 'height_variation_px': variation,
            'outside_texture_top': outside.get('top_ratio'), 'outside_texture_bottom': outside.get('bottom_ratio'),
            'outside_details': outside, 'consensus_total': consensus_total, 'consensus_ratio': ratio,
            'needs_review_reason': ';'.join(dict.fromkeys(reasons)),
            'quality_status': 'NEEDS_REVIEW' if reasons else ('RECOVERED' if recovered else 'PASS')}
